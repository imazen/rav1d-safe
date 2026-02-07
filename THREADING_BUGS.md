# Threading Bug Fix - Partial Implementation

## Status: Partially Fixed

### What Was Fixed
✅ Worker thread handles now wrapped in `Option<JoinHandle<()>>`
✅ Threads are properly signaled to exit via `tell_worker_threads_to_die()`
✅ Threads exit cleanly when decoder is dropped  

### What Still Needs Work
⚠️ Threads are not explicitly joined - they exit but may not be joined synchronously
⚠️ Thread cleanup cannot be verified in tests due to async nature

## The Architecture Problem

Worker threads hold `Arc<Rav1dContext>` references. When the decoder drops:
1. Main Arc in Decoder is dropped
2. Worker threads see die flag and exit
3. **Last worker thread** drops the final Arc<Rav1dContext>
4. If we implement `Drop` for Rav1dContext, it runs from within that worker thread
5. **Self-join deadlock**: Thread cannot join itself

## Attempted Solutions

### Attempt 1: Simple Drop with Join
**Failed:** Deadlock when last worker tries to join itself

### Attempt 2: Check for Self-Join
**Failed:** Last thread can't join itself, so it stays detached

### Attempt 3: Skip Join if Dropping from Worker  
**Failed:** All threads leak because we never join when dropping from worker context

## Root Cause

The fundamental issue: **Arc<Rav1dContext> is shared with worker threads.**

When the last Arc holder is a worker thread, that thread triggers Drop while still running.
This is a chicken-and-egg problem that can't be solved with the current architecture.

## Proper Solution (Requires Refactoring)

Move JoinHandles OUT of `Arc<Rav1dContext>`:

```rust
pub struct Decoder {
    ctx: Arc<Rav1dContext>,
    // Store handles separately so we can join from main thread
    worker_handles: Vec<JoinHandle<()>>,
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // Signal threads to die
        self.ctx.tell_worker_threads_to_die();
        
        // Join all workers synchronously (we're on main thread)
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
        
        // Now drop the Arc (workers have exited, no more Arc holders)
    }
}
```

This requires:
- Extracting JoinHandles from Rav1dContext during Decoder creation
- Storing them in Decoder instead
- Modifying Rav1dContextTaskType to NOT own handles

## Current Behavior

**Without fix:**
- Threads detach immediately when JoinHandles drop
- Threads keep running after decoder destroyed
- Resource leak

**With current partial fix:**
- Threads are signaled to exit cleanly
- Threads exit their main loop
- Threads are still detached (not joined), but exit promptly
- Much better than before, but not perfect

## Workaround

Use single-threaded mode to avoid the issue:

```rust
let decoder = Decoder::new()?;  // threads: 1, no workers
```

## Next Steps

1. **Short term:** Document the limitation
2. **Medium term:** Implement proper JoinHandle extraction architecture
3. **Long term:** Consider redesigning thread ownership model

## Files Modified

- `src/internal.rs`: Changed `Worker(JoinHandle<()>)` to `Worker(Option<JoinHandle<()>>)`
- `src/lib.rs`: Wrapped handle in `Some()` during thread spawn, made `tell_worker_threads_to_die` pub(crate)
- `tests/thread_cleanup_test.rs`: Added tests (currently some fail due to incomplete fix)

## Testing

Thread cleanup tests demonstrate the issue but don't all pass:
```bash
cargo test --test thread_cleanup_test
# Some tests fail - expected until proper refactoring is done
```
