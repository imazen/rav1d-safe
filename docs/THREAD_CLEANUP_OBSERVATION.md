# Thread cleanup startup observation

The native ARM full suite failed `test_multi_threaded_cleanup` after observing
two named workers immediately after creating a decoder configured for four.
The core creates four join handles. Rust sets the OS thread name inside the
child during initialization, so returning from spawn does not ensure that
`/proc/self/task/*/comm` already contains all four names.
[The Rust thread lifecycle source](https://doc.rust-lang.org/src/std/thread/lifecycle.rs.html)
shows that ordering. Startup scheduling is the inferred cause of the original
observation; the test counts names rather than join handles.

The user approved a bounded observation wait: poll names at one-millisecond
intervals for up to one second, then retain the existing `workers >= 4`
assertion. The post-drop zero-worker assertion also stays unchanged.
`just test-thread-start` runs all cleanup tests plus the existing 4K stress
fixture and repeats the four-worker test 100 times, each in its own process.
It requires `test-vectors/bench/photo_4k.avif`; missing data still fails.

On Neoverse-N1 with Rust 1.99.0 and archmage 0.9.30, all seven focused tests
and 100/100 startup repetitions pass. Deliberately configuring only three
workers fails after the one-second wait with the unchanged four-worker
assertion. This proves the wait cannot hide a missing worker.
[Raw logs and provenance](../benchmarks/thread_start_observation_2026-10-08/meta.json)
record commands, source hash and resource measurements. The approved run peaks
at 0.75 GiB RSS; the failing mutation peaks at 0.73 GiB.

Restoring the approved four-worker source passes all seven focused tests again,
plus one selected startup repetition. That wrapped run peaks at 0.75 GiB RSS.
