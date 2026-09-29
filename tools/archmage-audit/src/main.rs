//! archmage-audit: static analysis of archmage token-dispatch topology.
//!
//! Maps every function's compile-time feature context (vanilla | v1..v4 |
//! neon | wasm128 | scalar), classifies every call edge (boundary crossing,
//! in-context, re-dispatch, violation), and reports:
//!
//!   * `summon()` / `from_context()` / `incant!` usage per function
//!   * functions that should be `#[rite]` but are `#[arcane]` (trampoline waste)
//!   * `summon()` inside a non-vanilla context (redundant runtime detection)
//!   * `.unwrap()`/`.expect()` on tokens outside FFI wrappers
//!   * suffix-convention mismatches (`incant!` resolves `f` + `_v3`, so
//!     `f_avx2_safe` is invisible to it)
//!
//! Usage: archmage-audit <roots...> [--json] [--lint] [--tree <fn>]

use quote::ToTokens;
use std::collections::BTreeMap;
use std::path::PathBuf;
use syn::spanned::Spanned;
use syn::visit::Visit;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Ctx {
    Vanilla,
    Tier(String), // "v1".."v4", "neon", "wasm128", "scalar", "avx2", "avx512" (raw target_feature)
}

impl Ctx {
    fn label(&self) -> String {
        match self {
            Ctx::Vanilla => "vanilla".into(),
            Ctx::Tier(t) => t.clone(),
        }
    }
    /// Can a body in this context mint a proof for `other` via from_context()?
    /// Approximation of the archmage tier DAG: v4 covers v3,v2,v1; v3 covers v2,v1; etc.
    fn covers(&self, other: &Ctx) -> bool {
        fn rank(t: &str) -> u32 {
            match t {
                "v1" | "sse" | "sse2" => 1,
                "v2" | "sse4" | "sse4.1" | "sse4.2" => 2,
                "v3" | "avx2" => 3,
                "v4" | "avx512" | "avx512x" => 4,
                "neon" | "wasm128" | "scalar" => 0,
                _ => 0,
            }
        }
        match (self, other) {
            (Ctx::Tier(a), Ctx::Tier(b)) => {
                let (ra, rb) = (rank(a), rank(b));
                // cross-ISA never covers; scalar is covered by everything
                if b == "scalar" {
                    return true;
                }
                if (a.starts_with('v') || rank(a) > 0) && (b.starts_with('v') || rank(b) > 0) {
                    return ra >= rb && rb > 0;
                }
                a == b
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Inline {
    None,
    Hint,   // #[inline]
    Always, // #[inline(always)]
    Never,  // #[inline(never)]
}

#[derive(Debug)]
struct FnInfo {
    name: String,
    file: PathBuf,
    line: usize,
    end_line: usize,
    body_stmts: usize,
    inline: Inline,
    ctx: Ctx,
    is_entry: bool, // #[arcane]/#[autoversion]/#[magetypes] outer — callable from vanilla
    is_extern_c: bool, // unsafe extern "C" boundary
    is_test_cfg: bool,
    generics: String,
    token_param: Option<String>, // declared token param type, if any (Desktop64, X64V3Token, …)
    takes_fnptr: bool,           // any `fn(…)` fn-pointer parameter (DSP-table style)
    summons: u32,                // Token::summon() / summon_*() call sites in body
    summon_targets: Vec<String>, // tiers each summon targets (best-effort)
    from_context: u32,
    incants: u32,                   // incant! invocations
    local_macro_calls: Vec<String>, // arcane!(f) / scalar!(f) callees
    calls: Vec<String>,             // resolved-name candidates
    token_unwraps: u32,             // unwrap/expect on a token acquisition
    allows: Vec<String>,            // `// audit:allow(<kind>)` suppressions inside this fn
}

#[derive(Debug, Default)]
struct FnVisitor {
    fns: Vec<FnInfo>,
    in_test_mod: bool,
    allows: Vec<(usize, String)>, // (line, lint-kind) from `// audit:allow(<kind>)`
}

fn attr_tokens(attr: &syn::Attribute) -> String {
    attr.meta.to_token_stream().to_string()
}

fn classify_attrs(attrs: &[syn::Attribute]) -> (Ctx, bool, bool) {
    let mut ctx = Ctx::Vanilla;
    let mut is_entry = false;
    let mut is_test = false;
    for a in attrs {
        let t = attr_tokens(a);
        let compact: String = t.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.starts_with("cfg(") && compact.contains("test") {
            is_test = true;
        }
        if compact.contains("arcane") && !compact.contains("rite") {
            // #[arcane] / #[arcane(v3)] — entry point; context = declared tier or infer from token type later
            is_entry = true;
            ctx = tier_from_str(&compact).unwrap_or(Ctx::Tier("entry".into()));
        }
        if compact.contains("rite") {
            ctx = tier_from_str(&compact).unwrap_or(Ctx::Tier("token-param".into()));
        }
        if compact.contains("autoversion") {
            is_entry = true;
            ctx = tier_from_str(&compact).unwrap_or(Ctx::Tier("autoversion".into()));
        }
        if compact.contains("magetypes") {
            is_entry = true;
            ctx = Ctx::Tier("magetypes".into());
        }
        if compact.starts_with("target_feature") || compact.contains("target_feature(enable") {
            ctx = tier_from_str(&compact).unwrap_or(Ctx::Tier("target_feature".into()));
        }
    }
    (ctx, is_entry, is_test)
}

fn tier_from_str(s: &str) -> Option<Ctx> {
    for t in ["v4", "v3", "v2", "v1", "neon", "wasm128", "scalar"] {
        if s.contains(&format!("\"{t}\""))
            || s.contains(&format!("({t})"))
            || s.contains(&format!("({t},"))
            || s.contains(&format!(",{t}"))
            || s.contains(&format!("({t},"))
        {
            return Some(Ctx::Tier(t.into()));
        }
    }
    if s.contains("avx512") {
        Some(Ctx::Tier("avx512".into()))
    } else if s.contains("avx2") {
        Some(Ctx::Tier("avx2".into()))
    } else if s.contains("sse4") {
        Some(Ctx::Tier("sse4".into()))
    } else {
        None
    }
}

/// Scan a fn body for calls, summons, incants, local-macro dispatches.
/// Tier implied by a token type name (Desktop64 = x86-64-v3, etc.)
fn token_tier(type_str: &str) -> Option<&'static str> {
    let s = type_str;
    if s.contains("Desktop64") || s.contains("X64V3Token") {
        Some("v3")
    } else if s.contains("Server64") || s.contains("X64V4Token") {
        Some("v4")
    } else if s.contains("X64V2Token") {
        Some("v2")
    } else if s.contains("X64V1Token") {
        Some("v1")
    } else if s.contains("Neon") || s.contains("Arm64") {
        Some("neon")
    } else if s.contains("Wasm128") {
        Some("wasm128")
    } else if s.contains("Scalar") {
        Some("scalar")
    } else {
        None
    }
}

/// Tier targeted by a `summon_*` call, if identifiable.
fn summon_target_tier(call: &str) -> Option<&'static str> {
    match call {
        "summon_avx512" | "summon_avx512x" => Some("v4"),
        "summon_avx2" => Some("v3"),
        "summon_wasm128" => Some("wasm128"),
        _ => None, // bare Token::summon() — receiver type unknown here
    }
}

struct BodyScan<'a> {
    info: &'a mut FnInfo,
}

impl<'a> Visit<'a> for BodyScan<'a> {
    fn visit_expr_call(&mut self, node: &'a syn::ExprCall) {
        if let syn::Expr::Path(p) = &*node.func {
            let name = p
                .path
                .segments
                .last()
                .map(|s| s.ident.to_string())
                .unwrap_or_default();
            if name == "summon" || name.starts_with("summon_") {
                self.info.summons += 1;
                if let Some(t) = summon_target_tier(&name) {
                    self.info.summon_targets.push(t.to_string());
                } else {
                    // bare summon() — receiver type text
                    let recv = node.func.to_token_stream().to_string();
                    if let Some(t) = token_tier(&recv) {
                        self.info.summon_targets.push(t.to_string());
                    }
                }
            } else if name == "from_context" {
                self.info.from_context += 1;
            } else if !name.is_empty() {
                self.info.calls.push(name);
            }
        }
        syn::visit::visit_expr_call(self, node);
    }
    fn visit_expr_method_call(&mut self, node: &'a syn::ExprMethodCall) {
        let name = node.method.to_string();
        if matches!(name.as_str(), "unwrap" | "expect") {
            // only count unwraps whose receiver chain mentions a token/summon
            let recv = node.receiver.to_token_stream().to_string();
            if recv.contains("summon") || recv.contains("Token") || recv.contains("token") {
                self.info.token_unwraps += 1;
            } else {
                self.info.calls.push(name);
            }
        } else {
            self.info.calls.push(name);
        }
        syn::visit::visit_expr_method_call(self, node);
    }
    fn visit_macro(&mut self, node: &'a syn::Macro) {
        let mac = node
            .path
            .segments
            .last()
            .map(|s| s.ident.to_string())
            .unwrap_or_default();
        match mac.as_str() {
            "incant" => {
                self.info.incants += 1;
                // first arg = callee path
                let src = node.tokens.to_string();
                let callee = src
                    .split(['(', ' ', ':'])
                    .next()
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !callee.is_empty() {
                    self.info.calls.push(callee);
                }
            }
            "arcane" | "scalar" | "rite_call" => {
                // local dispatch macros: arcane!(f) / scalar!(f)
                let callee = node.tokens.to_string().trim().to_string();
                self.info.local_macro_calls.push(callee.clone());
                self.info.calls.push(callee);
            }
            _ => {}
        }
        syn::visit::visit_macro(self, node);
    }
}

impl<'ast> Visit<'ast> for FnVisitor {
    fn visit_item_mod(&mut self, node: &'ast syn::ItemMod) {
        let was = self.in_test_mod;
        let cfgd = node.attrs.iter().any(|a| {
            let c: String = attr_tokens(a)
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            // cfg(test), cfg(all(test,…)), cfg(any(test,…)), etc.
            c.starts_with("cfg(") && c.contains("test")
        });
        if cfgd {
            self.in_test_mod = true;
        }
        syn::visit::visit_item_mod(self, node);
        self.in_test_mod = was;
    }
    fn visit_item_fn(&mut self, node: &'ast syn::ItemFn) {
        self.record_fn(&node.sig, &node.attrs, &node.block);
        syn::visit::visit_item_fn(self, node);
    }
    fn visit_impl_item_fn(&mut self, node: &'ast syn::ImplItemFn) {
        self.record_fn(&node.sig, &node.attrs, &node.block);
        syn::visit::visit_impl_item_fn(self, node);
    }
}

impl FnVisitor {
    fn record_fn(&mut self, sig: &syn::Signature, attrs: &[syn::Attribute], block: &syn::Block) {
        let (mut ctx, is_entry, mut is_test_cfg) = classify_attrs(attrs);
        // #[test] on the fn itself, or we're inside a cfg(test) module
        is_test_cfg = is_test_cfg
            || self.in_test_mod
            || attrs.iter().any(|a| a.path().is_ident("test"))
            || sig.ident.to_string().starts_with("test_");
        // token param + fn-ptr detection across all params
        let mut token_param = None;
        let mut takes_fnptr = false;
        for arg in sig.inputs.iter() {
            if let syn::FnArg::Typed(pt) = arg {
                let ts = pt.ty.to_token_stream().to_string();
                if token_param.is_none() {
                    if let Some(_t) = token_tier(&ts) {
                        token_param = Some(ts.clone());
                    }
                }
                // fn-ptr types: `fn (…)`, `unsafe fn`, `extern "C" fn`, or
                // *_fn_t typedef names used by the DSP tables
                if ts.contains("fn (")
                    || ts.contains("fn(")
                    || ts.contains("unsafe fn")
                    || ts.ends_with("_fn_t")
                {
                    takes_fnptr = true;
                }
            }
        }
        // context = declared tier; else infer from token param type
        if let Ctx::Tier(t) = &ctx {
            if (t == "entry" || t == "token-param") && token_param.is_some() {
                if let Some(tier) = token_tier(token_param.as_ref().unwrap()) {
                    ctx = Ctx::Tier(tier.into());
                }
            }
        }
        let is_extern_c = sig
            .abi
            .as_ref()
            .is_some_and(|a| a.name.as_ref().is_some_and(|n| n.value() == "C"));
        let inline = attrs.iter().fold(Inline::None, |acc, a| {
            let c: String = attr_tokens(a)
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            if c.contains("inline(always)") {
                Inline::Always
            } else if c.contains("inline(never)") {
                Inline::Never
            } else if c.contains("inline") && acc == Inline::None {
                Inline::Hint
            } else {
                acc
            }
        });
        let start = sig.ident.span().start().line;
        let end = block.span().end().line;
        let mut info = FnInfo {
            name: sig.ident.to_string(),
            file: PathBuf::new(),
            line: start,
            end_line: end,
            body_stmts: block.stmts.len(),
            inline,
            ctx,
            is_entry,
            is_extern_c,
            is_test_cfg,
            generics: sig.generics.to_token_stream().to_string(),
            token_param,
            takes_fnptr,
            summons: 0,
            summon_targets: Vec::new(),
            from_context: 0,
            incants: 0,
            local_macro_calls: Vec::new(),
            calls: Vec::new(),
            token_unwraps: 0,
            allows: self
                .allows
                .iter()
                .filter(|(l, _)| *l >= start.saturating_sub(1) && *l <= end + 1)
                .map(|(_, k)| k.clone())
                .collect(),
        };
        let mut scan = BodyScan { info: &mut info };
        scan.visit_block(block);
        self.fns.push(info);
    }
}

fn tier_suffix(name: &str) -> &'static str {
    for suf in [
        "_avx512_safe",
        "_avx512_inner",
        "_avx512",
        "_avx2_safe",
        "_avx2_inner",
        "_avx2",
        "_sse4",
        "_v4",
        "_v3",
        "_v2",
        "_v1",
        "_neon",
        "_wasm128",
        "_scalar",
        "_inner",
    ] {
        if name.ends_with(suf) {
            return suf;
        }
    }
    ""
}

fn family(name: &str) -> &str {
    let suf = tier_suffix(name);
    if suf.is_empty() {
        name
    } else {
        &name[..name.len() - suf.len()]
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let json = args.iter().any(|a| a == "--json");
    let lint_only = args.iter().any(|a| a == "--lint");
    let tree_at = args
        .iter()
        .position(|a| a == "--tree")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let roots: Vec<PathBuf> = args
        .iter()
        .filter(|a| !a.starts_with('-') && Some(*a) != tree_at.as_ref())
        .map(PathBuf::from)
        .collect();

    let mut files = Vec::new();
    for root in &roots {
        for e in walkdir::WalkDir::new(root)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if e.path().extension().is_some_and(|x| x == "rs") {
                files.push(e.path().to_path_buf());
            }
        }
    }

    let mut all: Vec<FnInfo> = Vec::new();
    for f in &files {
        let src = match std::fs::read_to_string(f) {
            Ok(s) => s,
            Err(_) => continue,
        };
        let parsed = match syn::parse_file(&src) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let mut v = FnVisitor::default();
        // `// audit:allow(<kind>)` — suppression pragmas, line-scoped to the fn
        for (i, line) in src.lines().enumerate() {
            if let Some(pos) = line.find("audit:allow(") {
                let rest = &line[pos + 12..];
                let kind: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '-')
                    .collect();
                if !kind.is_empty() {
                    v.allows.push((i + 1, kind));
                }
            }
        }
        v.visit_file(&parsed);
        for info in &mut v.fns {
            info.file = f.clone();
        }
        all.extend(v.fns);
    }

    // name → defs index (simple name key; ambiguity counted, not resolved)
    let mut by_name: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, f) in all.iter().enumerate() {
        by_name.entry(f.name.clone()).or_default().push(i);
        // also index family base for suffix-variant resolution
    }

    // callee resolution: same-file def wins; else a globally unique name
    let resolve_callee = |caller: &FnInfo, name: &str| -> Option<usize> {
        let idxs = by_name.get(name)?;
        if let Some(&i) = idxs.iter().find(|&&i| all[i].file == caller.file) {
            return Some(i);
        }
        (idxs.len() == 1).then_some(idxs[0])
    };

    // caller index (for arcane-could-be-rite)
    let mut callers_of: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, f) in all.iter().enumerate() {
        for c in &f.calls {
            if let Some(j) = resolve_callee(f, c) {
                if j != i {
                    callers_of.entry(j).or_default().push(i);
                }
            }
        }
    }

    // ---- violations ----
    #[derive(Debug)]
    struct Violation {
        kind: &'static str,
        at: String,
        detail: String,
    }
    let mut violations = Vec::new();

    for f in &all {
        if f.is_test_cfg {
            continue;
        }
        let loc = format!("{}:{}:{}", f.file.display(), f.line, f.name);
        // summon() targeting a tier the context already covers = redundant
        // runtime detection. Summoning a HIGHER tier (v3 ctx → summon_avx512)
        // is legitimate escalation, not a violation.
        if f.summons > 0 && !f.is_extern_c {
            if let Ctx::Tier(ctx_tier) = &f.ctx {
                let redundant: Vec<&String> = f
                    .summon_targets
                    .iter()
                    .filter(|t| f.ctx.covers(&Ctx::Tier((*t).clone())))
                    .collect();
                let escalations = f.summons as usize - redundant.len();
                if !redundant.is_empty() {
                    violations.push(Violation {
                        kind: "summon-covered-by-context",
                        at: loc.clone(),
                        detail: format!(
                            "{} summon(s) targeting {} inside {}-context — from_context() proves it",
                            redundant.len(), redundant.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(","), ctx_tier,
                        ),
                    });
                }
                if escalations > 0 && ctx_tier == "entry" {
                    // entry ctx without resolvable tier — manual check
                    violations.push(Violation {
                        kind: "summon-in-unknown-context",
                        at: loc.clone(),
                        detail: format!(
                            "{escalations} summon(s); ctx tier unknown (check manually)"
                        ),
                    });
                }
            }
        }
        // unwrap/expect on a token acquisition outside FFI boundary
        if f.token_unwraps > 0 && !f.is_extern_c {
            violations.push(Violation {
                kind: "token-unwrap",
                at: loc.clone(),
                detail: format!(
                    "{} token unwrap/expect — gate, don't assert",
                    f.token_unwraps
                ),
            });
        }
        // tier-boundary: a context fn calling a non-inlineable vanilla fn —
        // the callee runs scalar-compiled and can't inline into this context.
        // Inlineable helpers (#[inline]/small bodies) inherit the caller's
        // features and are fine; extern "C" and entry wrappers are exempt.
        if let Ctx::Tier(t) = &f.ctx {
            let allowed = f.allows.iter().any(|a| a == "tier-boundary");
            if !allowed && !f.is_extern_c {
                for c in &f.calls {
                    if let Some(j) = resolve_callee(f, c) {
                        let callee = &all[j];
                        let inlineable = matches!(callee.inline, Inline::Hint | Inline::Always)
                            || callee.body_stmts <= 4;
                        if callee.ctx == Ctx::Vanilla
                            && !callee.is_entry
                            && !callee.is_extern_c
                            && !callee.is_test_cfg
                            && !inlineable
                        {
                            violations.push(Violation {
                                kind: "tier-boundary",
                                at: loc.clone(),
                                detail: format!(
                                    "{t}-ctx calls vanilla {} ({} stmts, no inline) — scalar island; allow with // audit:allow(tier-boundary)",
                                    callee.name, callee.body_stmts,
                                ),
                            });
                        }
                    }
                }
            }
        }
    }

    // arcane-could-be-rite: an entry fn (arcane trampoline) whose resolved
    // callers are ALL inside a context — the safe wrapper is dead weight;
    // #[rite] + incant! at callers is the idiom. Needs the caller index.
    for (i, f) in all.iter().enumerate() {
        if f.is_test_cfg
            || f.is_extern_c
            || !f.is_entry
            || f.allows.iter().any(|a| a == "arcane-could-be-rite")
        {
            continue;
        }
        let Some(callers) = callers_of.get(&i) else {
            continue;
        };
        let in_ctx = callers
            .iter()
            .filter(|&&c| matches!(all[c].ctx, Ctx::Tier(_)))
            .count();
        let vanilla = callers.len() - in_ctx;
        if !callers.is_empty() && vanilla == 0 {
            violations.push(Violation {
                kind: "arcane-could-be-rite",
                at: format!("{}:{}", f.file.display(), f.name),
                detail: format!(
                    "{} callers all in-context — trampoline unused",
                    callers.len()
                ),
            });
        }
    }

    // suffix-convention scan: fns whose name carries a tier suffix that incant! can't resolve
    // (incant! wants _v3/_v4/_neon/_wasm128/_scalar; _avx2_safe is invisible to it)
    for f in &all {
        if f.is_test_cfg {
            continue;
        }
        let suf = tier_suffix(&f.name);
        if matches!(suf, "_avx2_safe" | "_avx512_safe" | "_sse4") && matches!(f.ctx, Ctx::Tier(_)) {
            violations.push(Violation {
                kind: "suffix-not-incant-resolvable",
                at: format!("{}:{}", f.file.display(), f.name),
                detail: format!("suffix {suf} — incant! resolves _v3/_v4/_neon/_scalar; rename or it can't be incanted"),
            });
        }
    }

    // ---- per-module summary ----
    let mut per_file: BTreeMap<String, (u32, u32, u32, u32, u32, u32)> = BTreeMap::new();
    for f in &all {
        let e = per_file.entry(f.file.display().to_string()).or_default();
        if f.is_entry {
            e.0 += 1; // entries (trampolines)
        }
        if f.ctx != Ctx::Vanilla && !f.is_entry {
            e.1 += 1; // context fns
        }
        if f.ctx == Ctx::Vanilla && !f.is_entry {
            e.2 += 1; // plain
        }
        e.3 += f.summons;
        e.4 += f.incants;
        e.5 += f.local_macro_calls.len() as u32;
    }

    if json {
        let out: Vec<serde_json::Value> = all
            .iter()
            .map(|f| {
                serde_json::json!({
                    "name": f.name, "file": f.file, "line": f.line, "end_line": f.end_line,
                    "body_stmts": f.body_stmts, "inline": format!("{:?}", f.inline),
                    "ctx": f.ctx.label(),
                    "entry": f.is_entry, "extern_c": f.is_extern_c, "test_cfg": f.is_test_cfg,
                    "generics": f.generics, "token_param": f.token_param,
                    "takes_fnptr": f.takes_fnptr, "summons": f.summons,
                    "from_context": f.from_context, "incants": f.incants,
                    "local_macro_calls": f.local_macro_calls, "calls": f.calls,
                    "family": family(&f.name),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
        return;
    }

    if !lint_only {
        println!(
            "{:<52} {:>6} {:>7} {:>6} {:>7} {:>7} {:>6}",
            "file", "entry", "ctx-fn", "plain", "summon", "incant", "mac!"
        );
        for (file, (entries, ctxs, plains, summons, incants, macros)) in &per_file {
            println!(
                "{:<52} {:>6} {:>7} {:>6} {:>7} {:>7} {:>6}",
                file, entries, ctxs, plains, summons, incants, macros
            );
        }
        println!();
    }

    // suppressed count for visibility (allows that matched a would-be lint)
    let suppressed: usize = all.iter().map(|f| f.allows.len()).sum();

    println!(
        "=== violations ({} total, {} pragma suppressions in tree) ===",
        violations.len(),
        suppressed
    );
    let mut kinds: BTreeMap<&'static str, u32> = BTreeMap::new();
    for v in &violations {
        *kinds.entry(v.kind).or_default() += 1;
        println!("  [{}] {} — {}", v.kind, v.at, v.detail);
    }
    println!("\n=== score ===");
    for (k, n) in kinds {
        println!("  {k}: {n}");
    }
    let boundary = violations
        .iter()
        .filter(|v| v.kind == "tier-boundary")
        .count();
    let tramp = violations
        .iter()
        .filter(|v| v.kind == "arcane-could-be-rite")
        .count();
    println!("  tier-boundary score: {boundary} (scalar islands; lower is better)");
    println!("  unused trampolines: {tramp}");

    // ---- call tree for a root fn ----
    if let Some(root) = tree_at {
        println!("\n=== call tree from {root} ===");
        fn resolve<'a>(
            by: &'a BTreeMap<String, Vec<usize>>,
            all: &'a [FnInfo],
            name: &str,
        ) -> Option<&'a FnInfo> {
            // try exact, then family+tier candidates
            if let Some(idxs) = by.get(name) {
                return idxs.first().map(|&i| &all[i]);
            }
            for suf in [
                "_v4",
                "_v3",
                "_avx512_safe",
                "_avx2_safe",
                "_scalar",
                "_inner",
                "_avx2",
            ] {
                if let Some(idxs) = by.get(&format!("{name}{suf}")) {
                    return idxs.first().map(|&i| &all[i]);
                }
            }
            None
        }
        let mut stack: Vec<(String, usize)> = vec![(root.clone(), 0)];
        let mut seen = std::collections::HashSet::new();
        while let Some((name, depth)) = stack.pop() {
            if depth > 6 || !seen.insert((name.clone(), depth)) {
                continue;
            }
            match resolve(&by_name, &all, &name) {
                Some(f) => {
                    let tag = match &f.ctx {
                        Ctx::Vanilla => String::new(),
                        Ctx::Tier(t) => format!(" [{t}]"),
                    };
                    let entry = if f.is_entry { " ⇐TRAMPOLINE" } else { "" };
                    let sum = if f.summons > 0 {
                        &format!(" ⇐{}×summon", f.summons)
                    } else {
                        ""
                    };
                    let tp = f
                        .token_param
                        .as_deref()
                        .map(|t| format!(" <{t}>"))
                        .unwrap_or_default();
                    println!(
                        "{}{}{}{}{}{}  {}",
                        "  ".repeat(depth),
                        f.name,
                        tag,
                        entry,
                        sum,
                        tp,
                        f.file.display()
                    );
                    for c in f.calls.iter().rev() {
                        stack.push((c.clone(), depth + 1));
                    }
                }
                None => {
                    println!("{}{}", "  ".repeat(depth), name);
                }
            }
        }
    }
}
