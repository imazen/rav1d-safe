# AV1 decode tiers on macOS ARM64, 2026-10-16

Source `444cb912`, M4 Pro. Command: `~/work/claudehints/scripts/run-heavy --mem 12G --jobs 4 -- just arm-tiers-macos`. The recipe ran `cargo bench -p rav1d-safe --bench tier_isolation -- --cpu-policy=unrestricted --format=llm`. Full output is `~/tmp/rav1d-arm-tiers.log` on this Mac. The scheduler was unrestricted. Each fixture was decoded with one decoder thread and film grain enabled; all 52 decoded frames had exact YUV plane equality across tiers before timing.

| IVF fixture | NEON mean | Forced scalar mean |
| --- | ---: | ---: |
| 8-bit data, 12 frames | 30.62 ms | 119.77 ms |
| 10-bit data, 10 frames | 7.54 ms | 29.20 ms |
| 12-bit data, 10 frames | 7.59 ms | 30.05 ms |
| 8-bit film grain, 10 frames | 14.98 ms | 32.99 ms |
| 10-bit film grain, 10 frames | 19.06 ms | 39.39 ms |

All five comparisons favor NEON on the measured fixtures. The 10 and 12-bit data groups had 25–28% CV. This measures the default AVIF decoder dependency in isolation, not complete AVIF container decode. No dispatch change is supported by these results.

`run-heavy`: peak RSS 0.76 GiB, minimum available memory 5279 MiB, peak load 2.64.
