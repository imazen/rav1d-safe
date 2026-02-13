# Context Handoff — msac SIMD Dispatch Fix

## Current State (commit f7f1929)

Working on fixing the msac SSE2 SIMD regression. The `#[arcane]` dispatch added in the original SSE2 port caused a +10.5% regression because `#[arcane]` creates a function call boundary that prevents LLVM from inlining. Each adapt4/adapt8/hi_tok call paid ~4ns overhead (2 atomic loads for summon + function call + register save/restore), millions of times per frame.

### What's done:
1. **Token caching in MsacContext** — `Option<Desktop64>` and `Option<Arm64>` fields added to `MsacContext`, initialized once in `new()` via `summon()`. Dispatch reads stored token instead of calling `summon()` per-call.
2. **Dispatch re-enabled** — adapt4/adapt8/hi_tok SSE2 and NEON dispatch restored using stored tokens.
3. **Builds pass** — default safe, unchecked, asm, aarch64 cross-compile all pass.

### What's NOT done (fix these):

1. **Remove `force_scalar` feature from cfg gates** — The `not(feature = "force_scalar")` in `#[cfg(...)]` on the SSE2/NEON functions and dispatch is WRONG. Archmage handles scalar fallback at RUNTIME via `summon()` returning `None`. The cfg should just be `#[cfg(all(not(asm_msac), target_arch = "x86_64"))]` with no `force_scalar`. The runtime `if let Some(token) = s.avx2_token` branch handles scalar fallback. **Archmage has a "disable all tokens" method** for forcing scalar at runtime - use that instead of custom feature flags.


2. **Change `#[arcane]` to `#[rite]` on SSE2/NEON functions** — The user wants `#[rite]` (adds `#[target_feature]` + `#[inline]`, no wrapper) instead of `#[arcane]` (creates wrapper function that prevents inlining). The functions are:
   - `rav1d_msac_decode_symbol_adapt4_sse2` (line ~610)
   - `rav1d_msac_decode_symbol_adapt8_sse2` (line ~712)
   - `rav1d_msac_decode_hi_tok_sse2` (line ~807)
   - `rav1d_msac_decode_symbol_adapt4_neon` (line ~1222)
   - `rav1d_msac_decode_symbol_adapt8_neon` (line ~1324)
   - `rav1d_msac_decode_hi_tok_neon` (line ~1438)

   Just change `#[arcane]` → `#[rite]` and add `rite` to the import.

   **CRITICAL**: `#[rite]` functions are safe `#[target_feature]` functions. Calling them from a non-target-feature context (like the adapt4 dispatch function) requires `unsafe` in Rust 1.85+. Since we're under `forbid(unsafe_code)`, this WON'T compile unless the caller also has `#[target_feature]`.

   The user's solution: "hoist higher" — put `#[arcane]` at a higher level in the call chain (e.g., decode_coefs, decode_tile) so the msac functions are called from a matching-feature context and `#[rite]` can inline. This is a larger refactor.

   **For now**: the practical approach may be to keep `#[arcane]` on adapt4/adapt8/hi_tok SSE2/NEON functions (the wrapper IS the inlining barrier, but at least the per-call summon() overhead is eliminated). Then "hoist higher" as a follow-up.

3. **Benchmark** — Need to verify the cached-token approach is faster than the branchless scalar (commit 346e603). Run:
   ```bash
   just profile-quick  # 100 iterations, compare checked/unchecked times
   ```
   The tango baseline was exported at `target/tango/tango_decode` from commit 346e603 (scalar-only). Compare with:
   ```bash
   just tango-compare  # A/B comparison against exported baseline
   ```

4. **Stashed changes** — There are stashed changes (`git stash list`) from a broken attempt using manual `#[target_feature(enable = "sse2")] unsafe fn`. Drop them: `git stash drop`.

## Key Archmage 0.6 Rules (DO NOT FORGET)

See `memory/archmage.md` for full reference. Summary:

- **`#[rite]` = default for SIMD functions.** Adds `#[target_feature]` + `#[inline]`, no wrapper. Inlines into matching-feature callers.
- **`#[arcane]` = entry points only.** Creates wrapper function. Does NOT inline. Use sparingly.
- **Both are safe.** NEVER add `unsafe fn`, `#[allow(unsafe_code)]`, or manual `#[target_feature]`.
- **Summon once, pass token through.** Store in struct, not per-call.
- **No SSE2-only token.** Use `Desktop64` (AVX2 superset). SSE2 intrinsics work fine under it.
- **archmage handles scalar fallback at runtime** via `summon()` returning `None`. Don't use compile-time `force_scalar` features.

## Files Modified

- `src/msac.rs` — MsacContext fields, dispatch logic
- `CLAUDE.md` — Updated archmage rules in "MANDATORY: Safe intrinsics strategy"
- `memory/archmage.md` — NEW: full archmage 0.6 reference
- `memory/MEMORY.md` — Added archmage section

## Previous Commits (this session)

- `f7f1929` — WIP: cached tokens in MsacContext, re-enabled dispatch
- `346e603` — Removed SSE2 dispatch, branchless scalar only (was the baseline)
- `2f88177` — Added tango benchmarks
- `a251b70` — Added partial_asm feature

## Branch State

On `main`, 47 commits ahead of origin. Not pushed.
