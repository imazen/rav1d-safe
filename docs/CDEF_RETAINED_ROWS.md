# Retained CDEF destination rows (experiment)

Missing: current-source fallback and feature gates, complete decoder validation
and matched-binary timing. An earlier retained-row fixture and three padding
gates passed; no performance benefit has been measured.

CDEF filters consume a separate padded source scratch. A private destination
helper tries one mutable exact-row record for blocks up to eight pixels wide
and eight rows, copies
into the existing compact scratch, and retains the record until write-back.
The ordinary helper remains the fallback when the rectangle declines.
Single-threaded and untracked components use the existing direct path.
The rectangle probe receives the original caller and the exact width/rows/stride.

The opt-in is limited to the x86 CDEF callers; other architectures retain
the existing storage variants and helpers. Applying it to MC could collide with source
reads during intrabc; changing that call path requires a separate proof.
The test holds a gap write, checks that actual destination writes conflict,
compares the whole plane after write-back, and covers signed strides, odd
widths, and 8/10/12-bit pixels.

The experiment now shares the reviewed archmage 0.9.30 MC candidate.
Current-source fallback, feature, full decoder and timing gates remain pending. Run
`just test-cdef-retained` before full decoder validation.

The reversed-row fixture now places its held gap byte in the next lower row.
Using `origin + width + 1` on a negative stride would put that byte above
the rectangle hull and would not detect an overly wide hull registration.
The revised retained fixture passed in the sized-gate run below; whole-plane
and conflict assertions remain unchanged.

Source review traces the reservation lifetime through the x86 filter callback.
Padding finishes before destination reservation; the callback consumes its
independent scratch and copied destination. `src/cdef_apply.rs` backs up
pre-filter neighboring pixels before each filter call and walks blocks in
order within a band. Task boundaries use saved line buffers. These source
observations explain the intended alias separation; they do not substitute
for the threaded corpus and overlap gates.

`just check-cdef-retained-features` checks both-mode all-target lint and
ARM/WASM/C-FFI/ASM compilation. The picture helper compiles outside the
`safe_simd` module, which is excluded under ASM, so that boundary needs its
own compile check. These checks remain unrun for this experiment.

The first compiled retained-row fixture had only sixteen picture rows. Its
eight-row rectangle crossed more than the existing four-block registration
cap, so the helper correctly selected the ordinary compact fallback and the
retained-path assertion failed. The fast-path fixture now uses 256 picture
rows to arm the existing four-rows-per-block layout, keeping every shape,
stride sign, pixel comparison and conflict assertion unchanged. No tracker
cap is raised. This fixture passed with both stride signs and all three depths.


## Measured fixture gates

The [first fixture run](../benchmarks/cdef_retained_initial_gate_2026-10-08.log)
failed its retained-storage assertion on the sixteen-row plane, then aborted
when fallback write-back encountered the still-held conflict probe. Three
independent padding tests passed. The scope returned rc=100 after fourteen
seconds, peak RSS 1.56 GiB, minimum available 24,443 MiB, peak load 1.94.
The adaptive rectangle API may decline; callers must preserve compact fallback.

The [sized fixture run](../benchmarks/cdef_retained_sized_gate_2026-10-08.log)
passed the retained fixture and all three padding tests. The subsequent
all-target lint stopped at an explicit lifetime in the inherited MC oracle;
feature checks did not complete. This whole scope returned rc=101 after
24 seconds, peak RSS 1.62 GiB, minimum available 24,168 MiB, peak load 1.56.
The current shared MC oracle elides that lifetime and has its own passing
lint proof; this does not substitute for the retained experiment's feature gate.

A separate new test deliberately uses sixteen rows to exercise the declined
rectangle fallback, without provoking the retained-path conflict assertion.
It holds an inter-row gap, checks compact storage, changes every destination
pixel and compares the complete plane after write-back at signed strides and
8/10/12 bits. This added fallback test has not run. The production experiment
remains unpublished; the historical fixture logs describe their executed
snapshots, not a complete gate on the current source.
