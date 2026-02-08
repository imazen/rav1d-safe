# Fixed Bugs

Bugs found and fixed in rav1d-safe, with reproduction details.

---

## 1. msac AVX2 symbol_adapt16: unclamped return value and undersized MIN_PROB table

**Commit:** 278aa77
**Date:** 2026-02-05
**Severity:** Critical (decode crash)
**Origin:** Introduced during safe Rust SIMD port. NOT a bug in original dav1d.

### Symptoms

Decoding AV1 bitstreams without the `asm` feature panics with:
- `src/decode.rs:3492: valid variant` — `BlockPartition::from_repr()` returns `None` because `rav1d_msac_decode_symbol_adapt16` returned 16, which exceeds the valid range 0..=9.
- `src/ipred_prepare.rs:217: index out of bounds: the len is 14 but the index is 15` — downstream corruption from the invalid partition value.

### Reproduction

```bash
# Obtain test vectors (any AV1 IVF file triggers this)
# Example: av1-I-frame-320x240-agtm.ivf from chromium test data

# Fails (safe-simd, before fix):
RAV1D_TEST_IVF=/path/to/file.ivf \
  cargo test --lib --release --no-default-features --features "bitdepth_8,bitdepth_16" \
  -- decode_test_ivf --nocapture

# Passes (asm, reference):
RAV1D_TEST_IVF=/path/to/file.ivf \
  cargo test --lib --release --features "asm,bitdepth_8,bitdepth_16" \
  -- decode_test_ivf --nocapture
```

### Root Cause

Two bugs in `rav1d_msac_decode_symbol_adapt16_avx2` (`src/msac.rs`):

**Bug A: Unclamped trailing_zeros result**

The AVX2 code computes a movemask where bit `i` is set when `c >= v[i]` (the CDF probability threshold). It then uses `trailing_zeros()` to find the first set bit (first symbol where `c >= v`). When `c < v[i]` for ALL lanes — meaning the decoded symbol equals `n_symbols` — the mask is zero and `trailing_zeros()` returns 32, yielding `val = 32 >> 1 = 16`. This exceeds the valid variant range.

```rust
// BEFORE (broken):
let val = (mask.trailing_zeros() >> 1) as u8;

// AFTER (fixed):
let val = std::cmp::min((mask.trailing_zeros() >> 1) as u8, n_symbols);
```

Additionally, when `val == n_symbols`, the code accessed `v_arr[n_symbols]` for the renormalization value. While technically within the 16-element array bounds for `n_symbols < 16`, the value at that position was computed from garbage CDF data. The scalar fallback code computes v=0 in this case (the min_prob factor is `EC_MIN_PROB * (n - n) = 0`).

```rust
// BEFORE (broken):
let v_val = v_arr[val as usize] as u32;

// AFTER (fixed):
let v_val = if val >= n_symbols { 0u32 } else { v_arr[val as usize] as u32 };
```

**Bug B: MIN_PROB table too small**

The `MIN_PROB_16` table was 16 elements. The AVX2 code loads 16 values via `_mm256_loadu_si256` starting from offset `15 - n_symbols`:
- n_symbols=9 → offset=6, reads positions 6..22 (6 elements past end)
- n_symbols=3 → offset=12, reads positions 12..28 (12 elements past end)

This is undefined behavior (reading past array bounds).

```rust
// BEFORE (broken):
static MIN_PROB_16: [u16; 16] = [60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0];

// AFTER (fixed — padded to 31 entries):
static MIN_PROB_16: [u16; 31] = [
    60, 56, 52, 48, 44, 40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];
```

The same fix was applied to `MIN_PROB_16_ARM` for the NEON implementation.

### Why the original dav1d asm is not affected

The original x86 asm (`src/x86/msac.asm`) avoids both issues through its data layout:

1. **Table padding:** The `min_prob` table is immediately followed by `pw_0xff00: times 8 dw 0xff00` in the `.rodata` section. When SIMD loads read past `min_prob`, they get `0xff00` values, which produce very large `v` thresholds. This guarantees `c < v[i]` for unused lanes, leaving those mask bits as 0, so `tzcnt` ignores them naturally.

2. **Stack indexing:** The asm uses the raw byte offset from `tzcnt` to index the stack-stored `v` array. When `tzcnt = 32`, it reads `[buf+32]` which is past the 32-byte v array but still within the stack frame. The subsequent renormalization math (`u - v`, range update) handles whatever value is there because the asm code path for the CDF update also uses the mask to determine loop bounds correctly.

### Verification

After fix, both test vectors produce pixel-exact parity between asm and safe-simd:

```
av1-I-frame-320x240-agtm.ivf (1 frame, 320x240, 8-bit):
  ASM hash:       9dc180761aa7bc83c6db65b61e1bb274
  Safe-SIMD hash: 9dc180761aa7bc83c6db65b61e1bb274 ✓

test-25fps.av1.ivf (250 frames, 320x240, 8-bit):
  ASM hash:       67e102a51c14bf8815b6170c1993ebcf
  Safe-SIMD hash: 67e102a51c14bf8815b6170c1993ebcf ✓
```
