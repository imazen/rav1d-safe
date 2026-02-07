# Thread Joining Fix - Implementation Handoff

## OBJECTIVE
Fix worker thread joining in rav1d-safe so threads are properly cleaned up when decoder is dropped.

## THE PROBLEM

**Current architecture flaw:**
- `Arc<Rav1dContext>` contains `Vec<Rav1dContextTaskThread>` which contains `JoinHandle<()>`
- Worker threads ALSO hold `Arc<Rav1dContext>` references
- When decoder drops, last worker thread owns the last Arc
- That worker thread triggers `Drop` for Rav1dContext
- **Cannot join self → deadlock or leak**

## THE SOLUTION

**Move JoinHandles OUT of Arc<Rav1dContext> and into Decoder**

### Architecture Change

**Before:**
```
Decoder {
    ctx: Arc<Rav1dContext> {
        tc: Vec<Rav1dContextTaskThread> {
            task: Worker(JoinHandle)  ← PROBLEM: Inside Arc
        }
    }
}

Worker threads hold Arc<Rav1dContext> → circular ownership
```

**After:**
```
Decoder {
    ctx: Arc<Rav1dContext> {
        tc: Vec<Rav1dContextTaskThread> {
            task: Worker  ← No JoinHandle here
        }
    }
    handles: Vec<JoinHandle<()>>  ← JoinHandles HERE, outside Arc
}

Worker threads hold Arc<Rav1dContext> → no circular ownership with JoinHandles
```

## IMPLEMENTATION STEPS

### Step 1: Change Rav1dContextTaskType (src/internal.rs:344)

**Current:**
```rust
pub(crate) enum Rav1dContextTaskType {
    Worker(Option<JoinHandle<()>>),
    Single(Mutex<Box<Rav1dTaskContext>>),
}
```

**Change to:**
```rust
pub(crate) enum Rav1dContextTaskType {
    Worker,  // ← Remove JoinHandle completely
    Single(Mutex<Box<Rav1dTaskContext>>),
}
```

### Step 2: Return JoinHandles from rav1d_open (src/lib.rs)

**Current signature:**
```rust
pub(crate) fn rav1d_open(s: &Rav1dSettings) -> Result<Arc<Rav1dContext>, ()>
```

**Change to:**
```rust
pub(crate) fn rav1d_open(
    s: &Rav1dSettings
) -> Result<(Arc<Rav1dContext>, Vec<JoinHandle<()>>), ()>
```

**Implementation changes in rav1d_open (~line 240-260):**

```rust
// Current code:
let tc: Box<[Rav1dContextTaskThread]> = (0..n_tc)
    .map(|n| {
        let task_thread = Arc::clone(&task_thread);
        let thread_data = Arc::new(Rav1dTaskContextTaskThread::new(task_thread));
        let thread_data_copy = Arc::clone(&thread_data);
        let task = if n_tc > 1 {
            let handle = thread::Builder::new()
                .name(format!("rav1d-worker-{n}"))
                .spawn(|| rav1d_worker_task(thread_data_copy))
                .unwrap();
            Rav1dContextTaskType::Worker(Some(handle))  // ← Remove this
        } else {
            Rav1dContextTaskType::Single(Mutex::new(Box::new(
                Rav1dTaskContext::new(thread_data_copy),
            )))
        };
        Rav1dContextTaskThread { task, thread_data }
    })
    .collect();

// Change to:
let mut join_handles = Vec::new();
let tc: Box<[Rav1dContextTaskThread]> = (0..n_tc)
    .map(|n| {
        let task_thread = Arc::clone(&task_thread);
        let thread_data = Arc::new(Rav1dTaskContextTaskThread::new(task_thread));
        let thread_data_copy = Arc::clone(&thread_data);
        let task = if n_tc > 1 {
            let handle = thread::Builder::new()
                .name(format!("rav1d-worker-{n}"))
                .spawn(|| rav1d_worker_task(thread_data_copy))
                .unwrap();
            join_handles.push(handle);  // ← Store handle separately
            Rav1dContextTaskType::Worker  // ← No handle in enum
        } else {
            Rav1dContextTaskType::Single(Mutex::new(Box::new(
                Rav1dTaskContext::new(thread_data_copy),
            )))
        };
        Rav1dContextTaskThread { task, thread_data }
    })
    .collect();

// At end of function, return both:
let c = Arc::new(Rav1dContext { /* ... */ });
Ok((c, join_handles))  // ← Return tuple
```

### Step 3: Update Decoder to store JoinHandles (src/managed.rs)

**Current:**
```rust
pub struct Decoder {
    ctx: Arc<Rav1dContext>,
}

impl Decoder {
    pub fn with_settings(settings: Settings) -> Result<Self> {
        let rav1d_settings: Rav1dSettings = settings.into();
        let ctx = crate::src::lib::rav1d_open(&rav1d_settings)
            .map_err(|_| Error::InitFailed)?;
        Ok(Self { ctx })
    }
}
```

**Change to:**
```rust
pub struct Decoder {
    ctx: Arc<Rav1dContext>,
    worker_handles: Vec<std::thread::JoinHandle<()>>,
}

impl Decoder {
    pub fn with_settings(settings: Settings) -> Result<Self> {
        let rav1d_settings: Rav1dSettings = settings.into();
        let (ctx, worker_handles) = crate::src::lib::rav1d_open(&rav1d_settings)
            .map_err(|_| Error::InitFailed)?;
        Ok(Self { ctx, worker_handles })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        // Signal workers to exit
        self.ctx.tell_worker_threads_to_die();
        
        // Join all worker threads synchronously
        // This is safe because:
        // 1. We're on the main thread (Decoder is not Send)
        // 2. Workers have been signaled to exit
        // 3. We own the JoinHandles, not the workers
        for handle in self.worker_handles.drain(..) {
            match handle.join() {
                Ok(()) => {},
                Err(e) => {
                    eprintln!("Worker thread panicked during shutdown: {:?}", e);
                }
            }
        }
        
        // Now drop the Arc<Rav1dContext>
        // Workers have exited, so we're likely the last Arc holder
    }
}
```

### Step 4: Update C FFI (src/lib.rs - dav1d_open)

**Current:**
```rust
#[cfg(feature = "c-ffi")]
pub unsafe extern "C" fn dav1d_open(
    c_out: Option<NonNull<Option<Dav1dContext>>>,
    s: Option<NonNull<Dav1dSettings>>,
) -> Dav1dResult {
    // ...
    let c = rav1d_open(&s)?;
    *c_out = Some(c.into());
    Ok(())
}
```

**Change to:**
```rust
#[cfg(feature = "c-ffi")]
pub unsafe extern "C" fn dav1d_open(
    c_out: Option<NonNull<Option<Dav1dContext>>>,
    s: Option<NonNull<Dav1dSettings>>,
) -> Dav1dResult {
    // ...
    let (c, handles) = rav1d_open(&s)?;
    
    // Store handles somewhere for C FFI to join later
    // OR: Spawn a detached "janitor" thread to join them
    // OR: Accept that C FFI doesn't properly join (document limitation)
    
    // For now, spawn janitor thread:
    if !handles.is_empty() {
        let ctx_clone = Arc::clone(&c);
        std::thread::spawn(move || {
            // Wait for die signal
            while !ctx_clone.tc.iter().all(|tc| {
                if let Rav1dContextTaskType::Single(_) = tc.task {
                    true
                } else {
                    tc.thread_data.die.get()
                }
            }) {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            
            // Join all handles
            for handle in handles {
                let _ = handle.join();
            }
        });
    }
    
    *c_out = Some(c.into());
    Ok(())
}
```

### Step 5: Remove Option wrapper from Worker variant

Since Worker no longer contains JoinHandle, remove the `Option<JoinHandle<()>>` we added earlier:

```rust
// In src/lib.rs, line 252:
// Change from:
Rav1dContextTaskType::Worker(Some(handle))
// To:
Rav1dContextTaskType::Worker
```

### Step 6: Update tell_worker_threads_to_die

Already done - it's `pub(crate)` and doesn't touch JoinHandles.

### Step 7: Run Tests

```bash
cargo test --no-default-features --features "bitdepth_8,bitdepth_16" --test thread_cleanup_test --release
```

All tests should pass, especially:
- `test_multi_threaded_cleanup` 
- `test_auto_detect_threads_cleanup`
- `test_multiple_decoder_cycles`

## VALIDATION

After implementation, these assertions must hold:

1. **No deadlocks:** Decoder can be dropped from any thread
2. **No leaks:** Thread count returns to baseline after decoder drop
3. **Synchronous join:** Decoder::drop blocks until all workers exit
4. **Panic propagation:** Worker panics are reported (eprintln in Drop)

## EDGE CASES

### Case 1: Decoder dropped without ever calling decode()
- Workers are spawned but never do work
- Should still join cleanly

### Case 2: Multiple Decoders simultaneously
- Each owns its own JoinHandles
- Should not interfere

### Case 3: Decoder dropped while decode in progress
- Workers signaled to die
- May be mid-frame
- Should exit gracefully (die flag checked in loop)

## FILES TO MODIFY

1. `src/internal.rs` - Remove JoinHandle from Worker variant
2. `src/lib.rs` - Return handles from rav1d_open, update C FFI
3. `src/managed.rs` - Store handles in Decoder, implement Drop
4. `tests/thread_cleanup_test.rs` - Verify tests pass

## TESTING CHECKLIST

- [ ] Single-threaded decoder works
- [ ] Multi-threaded decoder works
- [ ] Threads are joined on drop (count returns to baseline)
- [ ] No deadlocks
- [ ] Works with threads=0 (auto-detect)
- [ ] Works with threads=N (explicit count)
- [ ] Multiple create/drop cycles don't leak
- [ ] Can drop decoder without calling decode()
- [ ] Worker panics are reported

## ROLLBACK PLAN

If this doesn't work, can rollback by reverting to git state before these changes:
```bash
git diff src/internal.rs src/lib.rs src/managed.rs
git checkout src/internal.rs src/lib.rs src/managed.rs
```

Current changes are in: `feat/fully-safe-intrinsics` branch

## CURRENT STATE

**Partially implemented:**
- ✅ Worker(Option<JoinHandle<()>>) - JoinHandle wrapped in Option
- ✅ tell_worker_threads_to_die() is pub(crate)
- ✅ Tests created
- ❌ JoinHandles still inside Arc<Rav1dContext>
- ❌ No Drop on Decoder

**Next step:** Implement Step 1-7 above to complete the fix.

## SUCCESS CRITERIA

```rust
#[test]
fn test_threads_joined_synchronously() {
    let baseline = count_threads();
    
    {
        let decoder = Decoder::with_settings(Settings {
            threads: 4,
            ..Default::default()
        }).unwrap();
        
        assert!(count_threads() >= baseline + 4);
    } // ← Drop happens here
    
    // Immediately after drop (no sleep needed):
    assert_eq!(count_threads(), baseline);  // ← MUST PASS
}
```

This test MUST pass when the fix is complete.
