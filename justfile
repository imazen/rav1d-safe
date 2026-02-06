# rav1d-safe justfile

# Lint allows for the ported codebase
asm_allows := "-A dead-code -A unused-imports -A unpredictable-function-pointer-comparisons -A mismatched-lifetime-syntaxes"
safe_allows := "-A dead-code -A unused-imports -A unused-variables -A unused-mut -A unused-parens -A private-interfaces -A mismatched-lifetime-syntaxes"

# ── Build ───────────────────────────────────────────────────────────

# Build safe-simd (no asm)
build-safe:
    RUSTFLAGS="-D warnings {{safe_allows}}" cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16"

# Build with asm
build-asm:
    RUSTFLAGS="-D warnings {{asm_allows}}" cargo build --release --features "asm,bitdepth_8,bitdepth_16"

# Build safe-simd + c-ffi
build-cffi:
    RUSTFLAGS="-D warnings {{safe_allows}}" cargo build --release --no-default-features --features "bitdepth_8,bitdepth_16,c-ffi"

# ── Test ────────────────────────────────────────────────────────────

# Run unit tests (safe-simd)
test:
    RUSTFLAGS="-D warnings {{safe_allows}}" cargo test --lib --release --no-default-features --features "bitdepth_8,bitdepth_16"

# Run unit tests (asm)
test-asm:
    RUSTFLAGS="-D warnings {{asm_allows}}" cargo test --lib --release --features "asm,bitdepth_8,bitdepth_16"

# Run decode parity test with an IVF file
# Usage: just test-decode /path/to/file.ivf
test-decode ivf:
    RUSTFLAGS="{{safe_allows}}" RAV1D_TEST_IVF={{ivf}} cargo test --lib --release --no-default-features --features "bitdepth_8,bitdepth_16" -- decode_test_ivf --nocapture

# Run decode test with asm (reference)
test-decode-asm ivf:
    RUSTFLAGS="{{asm_allows}}" RAV1D_TEST_IVF={{ivf}} cargo test --lib --release --features "asm,bitdepth_8,bitdepth_16" -- decode_test_ivf --nocapture

# ── Lint ────────────────────────────────────────────────────────────

# Format check
fmt:
    cargo fmt --check

# Format fix
fmt-fix:
    cargo fmt

# Clippy (safe-simd)
clippy:
    cargo clippy --no-default-features --features "bitdepth_8,bitdepth_16" -- -D warnings {{safe_allows}}

# Clippy (asm)
clippy-asm:
    cargo clippy --features "asm,bitdepth_8,bitdepth_16" -- -D warnings {{asm_allows}}

# ── Cross ───────────────────────────────────────────────────────────

# Cross-check aarch64
check-aarch64:
    RUSTFLAGS="-D warnings {{safe_allows}}" cargo check --target aarch64-unknown-linux-gnu --no-default-features --features "bitdepth_8,bitdepth_16"

# ── CI (run everything) ────────────────────────────────────────────

# Run the full CI suite locally
ci: fmt clippy clippy-asm build-safe build-asm build-cffi test test-asm check-aarch64
