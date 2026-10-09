# Block initialization experiment

Missing: compile, codegen comparison, complete correctness gates and timing.
No speedup is claimed.

The independently rebuilt 5b51b144 untracked profile binary initializes its
stack Av1Block before checking frame_thread.pass. Its decode_b entry has five
initialization stores before the pass branch. Nonzero passes instead use the
stored block behind a mutex and do not need the stack default.

The experiment initializes the stack block only in the branch that borrows
it. Rust checks definite initialization; no unsafe or representation change
is introduced. The current source includes the reviewed MC pair-window follow-ups and
signed-source repair, using Rust 1.99.0 and archmage 0.9.30. The preserved
`f910b2f2` untracked binary has five stores before the pass branch; the `decode_b` symbol
is 57,224 bytes. The lead-added MC parity tests are also present. Static
stores and code size do not establish a throughput benefit.

Test with all 803 sidecar MD5s at 1/2/4/8 threads and delay0 in tracked and
untracked modes, the full nextest suites, and the generated thread/backpressure
vectors. Timing must include both modes, 8-bit and 10-bit streams and an A/A
control. Inspect the resulting entry code before crediting removed stores.

Include four-worker frame delay2 in the timing matrix. The default delay1
A/B is still a control, but the removed initialization belongs to nonzero
frame passes, which use the stored block. A delay1-only comparison cannot
establish the effect on those passes. Measure both delay settings and keep
their results distinct.

The current isolated child shares the signed warp destination repair with
the planned baseline. Independent review confirms that only the placement
of `Av1Block::default()` changes in decode glue: `Av1Block` is Copy, its
default constructs the intra variant, and the stack slot is borrowed only
in the branch that initializes it. The nonzero-pass branch still locks and
borrows the stored block. No CDEF line-buffer pitch hint is present in this
child. These are source-level facts; compilation, emitted stores and
measured frame-threaded throughput remain pending.
