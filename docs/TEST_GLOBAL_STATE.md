# Unit tests and process-wide state

Token permutations disable archmage tokens process-wide. Every token-dependent
ITX test in `part05_tests.rs` holds `token_test_lock`, including tests that
summon once to check availability and later summon again. The WHT permutation
sweep alone cannot protect other tests sharing its process.

The loop-filter window tests require a fresh tile-threading latch. They run
their existing assertions in a child process selected by the full test name.
The parent checks both successful exit and a unique completion marker, because
libtest exits successfully when a filter matches no tests. The deliberately
overlapping window still must fail inside the child.

The prepared tree passed all 115 library tests under nextest and again under
plain libtest with eight test threads on 2026-10-08. No assertion, threshold or
ignore changed. Resource record for the combined lint/compile/test gate:
rc=0, 42s, peak-RSS 1.50GiB, min-avail 25169MiB, peak-load 4.82.
