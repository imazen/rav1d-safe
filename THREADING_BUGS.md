# Threading Bugs Analysis - rav1d-safe

## Executive Summary

**Primary Issue:** Worker threads are never joined when the decoder is closed/dropped. This leads to:
- Detached threads continuing to run after decoder destruction
- Inability to verify clean shutdown in tests  
- Potential false positives in leak detection tools

## Critical Bug: No Thread Joining

### The Problem

When `Decoder::drop()` is called:
1. `Arc<Rav1dContext>` refcount decrements
2. On last reference: `rav1d_close()` is called (implicit via Arc drop)
3. `tell_worker_threads_to_die()` sets `die` flag and signals threads
4. **`JoinHandle`s are dropped WITHOUT calling `.join()`**
5. Threads become **detached** and continue running

### Evidence

```bash
$ rg "\.join\(\)" src/
# NO RESULTS - threads are never joined!
```

No Drop implementation exists for:
- `Rav1dContext`  
- `Rav1dContextTaskThread`

When `JoinHandle` is dropped without `.join()`, Rust detaches the thread.

### Code Locations

**Thread storage:**
```rust
// src/internal.rs:344
pub(crate) enum Rav1dContextTaskType {
    Worker(JoinHandle<()>),  // ← Never joined!
    Single(Mutex<Box<Rav1dTaskContext>>),
}
```

**Thread shutdown signal:**
```rust
// src/lib.rs:722
pub(crate) fn rav1d_close(c: Arc<Rav1dContext>) {
    rav1d_flush(c);
    c.tell_worker_threads_to_die();  // Sets die flag, notifies threads
    // JoinHandles drop here without join() - threads become detached!
}
```

**Worker thread main loop:**
```rust
// src/thread_task.rs:803
'outer: while !tc.task_thread.die.get() {
    // Process tasks...
}
// Thread exits normally when die flag is set
```

### Impact

1. **Threads may outlive the decoder:**
   - Decoder dropped, but threads keep running
   - Accessing freed resources (mitigated by Arc, see below)
   
2. **Cannot verify cleanup in tests:**
   - No way to assert all threads exited
   - Makes testing thread safety difficult

3. **Leak detection false positives:**
   - LeakSanitizer may report leaks from still-running threads
   - AddressSanitizer may report use-after-free (Arc prevents this)

4. **Resource exhaustion:**
   - Creating/dropping many decoders could exhaust thread limits
   - Threads don't exit promptly

### Arc Safety - NOT A BUG

The worker thread pattern IS safe due to Rust's temporary lifetime extension:

```rust
// src/thread_task.rs:779
let c = &*task_thread.c.lock().take().unwrap();
//         ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Temporary Arc
//      ^^ Reference extends Arc's lifetime
```

The Arc is kept alive for the duration of the function, preventing use-after-free.
Verified with test case - compiles and runs correctly.

## Secondary Issue: No Timeout on Thread Exit

### The Problem

Thread only checks `die` flag at loop start. If thread is:
- Blocked on condvar
- Stuck in long-running task
- Waiting on another thread

...it may not exit promptly.

### Fix Needed

Replace `cond.wait()` with `cond.wait_timeout()`:

```rust
let timeout = Duration::from_secs(1);
while let Ok(_) = ttd.cond.wait_timeout(lock, timeout)? {
    if tc.task_thread.die.get() {
        break;
    }
}
```

## Tertiary Issue: Panics Silently Lost

If worker thread panics, the panic is lost because `JoinHandle::drop()` doesn't check the result.

### Fix Needed

```rust
impl Drop for Rav1dContextTaskThread {
    fn drop(&mut self) {
        if let Rav1dContextTaskType::Worker(handle) = &mut self.task {
            match handle.join() {
                Ok(()) => {},
                Err(e) => eprintln!("Worker thread panicked: {:?}", e),
            }
        }
    }
}
```

## Recommended Fix

### Step 1: Implement Drop for Rav1dContext

```rust
impl Drop for Rav1dContext {
    fn drop(&mut self) {
        // Signal threads to exit
        self.tell_worker_threads_to_die();
        
        // Join all worker threads
        for tc in &mut self.tc {
            if let Rav1dContextTaskType::Worker(handle) = &mut tc.task {
                // Take ownership of the JoinHandle (replace with dummy)
                let handle = std::mem::replace(handle, /* need placeholder */);
                
                match handle.join() {
                    Ok(()) => {},
                    Err(e) => {
                        eprintln!("Worker thread panicked during shutdown: {:?}", e);
                    }
                }
            }
        }
    }
}
```

**Problem:** Can't take ownership of `JoinHandle` in Drop because it's behind `&mut self`.

**Better Solution:** Move `JoinHandle` to `Option<JoinHandle>`:

```rust
pub(crate) enum Rav1dContextTaskType {
    Worker(Option<JoinHandle<()>>),  // ← Wrap in Option
    Single(Mutex<Box<Rav1dTaskContext>>),
}

impl Drop for Rav1dContext {
    fn drop(&mut self) {
        self.tell_worker_threads_to_die();
        
        for tc in &mut self.tc {
            if let Rav1dContextTaskType::Worker(handle) = &mut tc.task {
                if let Some(handle) = handle.take() {
                    let _ = handle.join();  // Ignore panic payload
                }
            }
        }
    }
}
```

### Step 2: Add Thread Cleanup Test

```rust
#[test]
fn test_worker_thread_cleanup() {
    use std::time::Duration;
    
    fn count_threads() -> usize {
        #[cfg(target_os = "linux")]
        {
            std::fs::read_dir("/proc/self/task").unwrap().count()
        }
        #[cfg(not(target_os = "linux"))]
        {
            // Fallback: can't easily count threads on other platforms
            return 0;
        }
    }
    
    #[cfg(target_os = "linux")]
    {
        let baseline = count_threads();
        
        {
            let mut decoder = Decoder::with_settings(Settings {
                threads: 4,
                ..Default::default()
            }).unwrap();
            
            let _ = decoder.decode(&[]);
            
            let with_workers = count_threads();
            assert!(with_workers >= baseline + 4, 
                "Workers not spawned");
        }
        
        // Decoder dropped, threads should be joined
        std::thread::sleep(Duration::from_millis(50));
        
        let after_drop = count_threads();
        assert_eq!(baseline, after_drop,
            "Worker threads leaked: {} baseline, {} after drop",
            baseline, after_drop);
    }
}
```

## Impact on Managed API

The managed API is affected when using frame threading:

```rust
// Single-threaded - NO ISSUE (no workers spawned)
let decoder = Decoder::new()?;  // threads: 1

// Multi-threaded - AFFECTED (workers spawned but not joined)
let decoder = Decoder::with_settings(Settings {
    threads: 0,  // Auto-detect - spawns workers
    ..Default::default()  
})?;
```

Currently, multi-threaded decoders leak threads on drop.

## Testing Recommendations

1. Add thread cleanup test (see above)
2. Run with LeakSanitizer: `RUSTFLAGS="-Z sanitizer=leak" cargo +nightly test`
3. Run with AddressSanitizer: `RUSTFLAGS="-Z sanitizer=address" cargo +nightly test`
4. Stress test: Create/drop many decoders in a loop, verify thread count stable

## Priority

**High:** This should be fixed before any production use of multi-threaded decoding via the managed API.

**Workaround for now:** Use single-threaded decoding (`threads: 1`) to avoid the issue entirely.
