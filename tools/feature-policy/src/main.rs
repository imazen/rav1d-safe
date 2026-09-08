//! Conservative source gate: environment access needs a private feature or tests.
use quote::ToTokens;
use std::path::Path;
use syn::{
    Attribute, Item, Meta, Token,
    punctuated::Punctuated,
    visit::{self, Visit},
};

// Is this cfg definitely false with every private feature and cfg(test) off?
// Unknown platform/public-feature predicates stay unknown, including under not.
fn cfg_value(meta: &Meta) -> Option<bool> {
    match meta {
        Meta::Path(p) if p.is_ident("test") => Some(false),
        Meta::NameValue(n) if n.path.is_ident("feature") => {
            if let syn::Expr::Lit(v) = &n.value {
                if let syn::Lit::Str(s) = &v.lit {
                    if s.value().starts_with("__") {
                        return Some(false);
                    }
                }
            }
            None
        }
        Meta::List(l) => {
            let xs = l
                .parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)
                .ok()?;
            let vs: Vec<_> = xs.iter().map(cfg_value).collect();
            if l.path.is_ident("all") {
                if vs.contains(&Some(false)) {
                    Some(false)
                } else if vs.iter().all(|v| *v == Some(true)) {
                    Some(true)
                } else {
                    None
                }
            } else if l.path.is_ident("any") {
                if vs.contains(&Some(true)) {
                    Some(true)
                } else if vs.iter().all(|v| *v == Some(false)) {
                    Some(false)
                } else {
                    None
                }
            } else if l.path.is_ident("not") && vs.len() == 1 {
                vs[0].map(|v| !v)
            } else {
                None
            }
        }
        _ => None,
    }
}
fn gated(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|a| {
        a.path().is_ident("cfg")
            && a.parse_args::<Meta>().ok().as_ref().and_then(cfg_value) == Some(false)
    })
}
#[derive(Default)]
struct Audit {
    protected: bool,
    errors: Vec<String>,
    accesses: usize,
}
impl Audit {
    fn access(&mut self, text: String) {
        self.accesses += 1;
        if !self.protected {
            self.errors.push(text);
        }
    }
}
impl<'ast> Visit<'ast> for Audit {
    fn visit_file(&mut self, f: &'ast syn::File) {
        let old = self.protected;
        self.protected |= gated(&f.attrs);
        visit::visit_file(self, f);
        self.protected = old;
    }
    fn visit_item(&mut self, i: &'ast Item) {
        let attrs: &[Attribute] = match i {
            Item::Fn(x) => &x.attrs,
            Item::Mod(x) => &x.attrs,
            Item::Impl(x) => &x.attrs,
            Item::Trait(x) => &x.attrs,
            Item::Const(x) => &x.attrs,
            Item::Static(x) => &x.attrs,
            Item::Use(x) => &x.attrs,
            Item::Macro(x) => &x.attrs,
            _ => &[],
        };
        let old = self.protected;
        self.protected |= gated(attrs);
        visit::visit_item(self, i);
        self.protected = old;
    }
    fn visit_path(&mut self, p: &'ast syn::Path) {
        let names: Vec<_> = p.segments.iter().map(|s| s.ident.to_string()).collect();
        if names.iter().any(|s| s == "env")
            && names
                .last()
                .is_some_and(|s| ["var", "var_os", "vars", "vars_os"].contains(&s.as_str()))
        {
            self.access(p.to_token_stream().to_string());
        }
        if names.last().is_some_and(|s| {
            [
                "getenv",
                "_wgetenv",
                "GetEnvironmentVariableA",
                "GetEnvironmentVariableW",
            ]
            .contains(&s.as_str())
        }) {
            self.access(p.to_token_stream().to_string());
        }
        visit::visit_path(self, p);
    }
    fn visit_item_use(&mut self, u: &'ast syn::ItemUse) {
        // Also reject ungated aliases such as `use std::{env as e}` and
        // `use std::env::var as read`. Direct paths above cover env::var.
        let text = u.tree.to_token_stream().to_string();
        let words: Vec<_> = text
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .collect();
        if words.contains(&"std") && words.contains(&"env") {
            self.access(text);
        }
        visit::visit_item_use(self, u);
    }
    fn visit_macro(&mut self, m: &'ast syn::Macro) {
        let tokens = m.tokens.to_string();
        if m.path
            .segments
            .last()
            .is_some_and(|s| s.ident == "env" || s.ident == "option_env")
            || tokens.contains("std :: env")
        {
            self.access(m.to_token_stream().to_string());
        }
        visit::visit_macro(self, m);
    }
}
fn scan(path: &Path, files: &mut usize, accesses: &mut usize, errors: &mut Vec<String>) {
    if path.is_dir() {
        let mut paths: Vec<_> = std::fs::read_dir(path)
            .unwrap()
            .map(|x| x.unwrap().path())
            .collect();
        paths.sort();
        for p in paths {
            scan(&p, files, accesses, errors);
        }
    } else if path.extension().is_some_and(|x| x == "rs") {
        let source = std::fs::read_to_string(path).unwrap();
        let file = syn::parse_file(&source).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let mut audit = Audit::default();
        audit.visit_file(&file);
        *files += 1;
        *accesses += audit.accesses;
        errors.extend(
            audit
                .errors
                .iter()
                .map(|e| format!("{}: {e}", path.display())),
        );
    }
}
fn main() {
    let mut files = 0;
    let mut accesses = 0;
    let mut errors = Vec::new();
    for path in std::env::args().skip(1) {
        scan(Path::new(&path), &mut files, &mut accesses, &mut errors);
    }
    assert!(files > 0, "no source files scanned");
    for e in &errors {
        eprintln!("UNGATED environment access: {e}");
    }
    assert!(
        errors.is_empty(),
        "environment access must require a __feature (test-only code is excluded)"
    );
    println!(
        "Environment gate passed: {files} source files, {accesses} explicit accesses/imports/macros inspected"
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    fn rejects(source: &str) -> bool {
        let mut a = Audit::default();
        a.visit_file(&syn::parse_file(source).unwrap());
        !a.errors.is_empty()
    }
    #[test]
    fn admits_only_compile_time_private_gates() {
        for prefix in [
            "",
            "#[cfg(feature = \"asm\")]",
            "#[cfg(not(feature = \"__probe\"))]",
            "#[cfg(any(feature = \"__probe\", unix))]",
        ] {
            assert!(rejects(&format!(
                "{prefix} fn f() {{ std::env::var(\"X\"); }}"
            )));
        }
        for prefix in [
            "#[cfg(feature = \"__probe\")]",
            "#[cfg(all(feature = \"__probe\", unix))]",
            "#[cfg(test)]",
        ] {
            assert!(!rejects(&format!(
                "{prefix} fn f() {{ std::env::var(\"X\"); }}"
            )));
        }
        assert!(rejects("use std::{env as e}; fn f() { e::var(\"X\"); }"));
        assert!(rejects(
            "fn f() { if cfg!(feature = \"__probe\") { std::env::var(\"X\"); } }"
        ));
    }
}
