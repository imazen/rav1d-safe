
2026-02-08: User asked to clean up allow/deny/forbid gating. Made 5 changes.
## 2026-02-09: Feature-gate disjoint-mut cleanup
User requested cleanup of rav1d-disjoint-mut to be minimal and auditable.
User suspicious of StridedBuf/PicBuf raw pointers — "the whole point is we need lifetime enforcement for safety in rust."
Chose option 3: eliminate from_byte_slice. Implemented copy-based approach (from_slice_copy + copy_pixels_to).

## 2026-02-09: SIMD Module Bisection Testing

User requested systematic identification of buggy SIMD modules by enabling one at a time against a scalar baseline.

### Results (8-bit tests, 571 total):

| Module | Pass/Total | Status |
|--------|-----------|--------|
| Scalar baseline (force_scalar) | 569/571 | Reference |
| MC | 569/571 | CLEAN |
| ipred | 569/571 | CLEAN |
| filmgrain | 569/571 | CLEAN |
| pal | 569/571 | CLEAN |
| msac | 569/571 | CLEAN |
| cdef (with sub→add fix) | 568/571 | 1 extra failure |
| loopfilter | 435/571 | BUGGY (-134) |
| looprestoration | 215/571 | BUGGY (-354) |
| ITX | 30/571 | SEVERELY BUGGY (-539) |
| All SIMD enabled | 30/571 | Dominated by ITX bugs |

Known non-SIMD failures: annexb, section5 (both format issues, not decode bugs).

## 2026-02-09: ITX Bug Diagnosis

User asked to add SIMD-vs-scalar comparison diagnostic to ITX dispatch.
Found multiple bug categories in safe_simd/itx.rs:

1. **Coefficient reading order** (4x4 scalar functions): Read coeff[y*4+x] (row-major) instead of coeff[y+x*4] (column-major). Affects all 14 ADST/FLIPADST/hybrid 4x4 types.
2. **IDTX scaling** (8x8, 16x16 arcane): Computes (c*4+8)>>4 instead of correctly handling intermediate shift between row/col passes. Factor of ~2x error.
3. **Rectangular DCT_DCT** (R32x16, R16x4, R4x16 arcane): Off by up to 20 pixels. Root cause TBD.

## 2026-02-09: ITX Final Scaling Fix

User identified wrong final scaling in large ITX SIMD functions. The correct final scaling is ALWAYS (c + 8) >> 4 per the scalar reference (inv_txfm_add, src/itx.rs line 159). Several functions incorrectly used (c + 4) >> 3 (for 32x32, 16x32, 32x16) and (c + 2) >> 2 (for 64x64 and related). Fixed 11 locations across 8bpc and 16bpc variants.
