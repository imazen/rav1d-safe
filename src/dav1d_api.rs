//! C FFI wrappers for the dav1d/rav1d ABI.
//!
//! All 19 `dav1d_*` `extern "C"` entry points live here, gated behind
//! `#[cfg(feature = "c-ffi")]` at the module level (see crate root `lib.rs`).
//! The Rust core API (`rav1d_*`) stays in `src/lib.rs`.
//!
//! ## Symbol naming
//!
//! When `dav1d-compat` feature is enabled (default with `c-ffi`), symbols are
//! exported as `dav1d_*` for drop-in replacement of libdav1d.
//!
//! When `dav1d-compat` is disabled, symbols are exported as `rav1d_*`, allowing
//! side-by-side loading with real libdav1d for benchmarking.
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use crate::include::common::validate::validate_input;
use crate::include::dav1d::common::Dav1dDataProps;
use crate::include::dav1d::common::Rav1dDataProps;
use crate::include::dav1d::data::Dav1dData;
use crate::include::dav1d::dav1d::Dav1dContext;
use crate::include::dav1d::dav1d::Dav1dEventFlags;
use crate::include::dav1d::dav1d::Dav1dSettings;
use crate::include::dav1d::headers::Dav1dSequenceHeader;
use crate::include::dav1d::picture::Dav1dPicture;
use crate::include::dav1d::picture::Rav1dPicture;
use crate::src::c_arc::RawArc;
use crate::src::c_box::FnFree;
use crate::src::error::Dav1dResult;
use crate::src::error::Rav1dError::EINVAL;
use crate::src::error::Rav1dResult;
use crate::src::lib::rav1d_apply_grain;
use crate::src::lib::rav1d_close;
use crate::src::lib::rav1d_flush;
use crate::src::lib::rav1d_get_frame_delay;
use crate::src::lib::rav1d_get_picture;
use crate::src::lib::rav1d_open;
use crate::src::lib::rav1d_send_data;
use crate::src::lib::DAV1D_API_VERSION_MAJOR;
use crate::src::lib::DAV1D_API_VERSION_MINOR;
use crate::src::lib::DAV1D_API_VERSION_PATCH;
use crate::src::lib::DAV1D_VERSION;
use crate::src::obu::rav1d_parse_sequence_header;
use crate::src::send_sync_non_null::SendSyncNonNull;
use std::ffi::c_char;
use std::ffi::c_uint;
use std::ffi::c_void;
use std::mem;
use std::ptr;
use std::ptr::NonNull;
use std::slice;
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use to_method::To as _;

use crate::include::dav1d::data::Rav1dData;
use crate::include::dav1d::dav1d::Rav1dSettings;
use crate::src::internal::Rav1dContextTaskType;

#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_version")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_version")]
#[cold]
pub extern "C" fn dav1d_version() -> *const c_char {
    DAV1D_VERSION.as_ptr()
}

/// Get the `dav1d` library C API version.
///
/// Return a value in the format `0x00XXYYZZ`, where `XX` is the major version,
/// `YY` the minor version, and `ZZ` the patch version.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_version_api")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_version_api")]
#[cold]
pub extern "C" fn dav1d_version_api() -> c_uint {
    u32::from_be_bytes([
        0,
        DAV1D_API_VERSION_MAJOR,
        DAV1D_API_VERSION_MINOR,
        DAV1D_API_VERSION_PATCH,
    ])
}

/// # Safety
///
/// * `s` must be valid to [`ptr::write`] to.
///   The former contents of `s` are not [`drop`]ped and it may be uninitialized.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_default_settings")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_default_settings")]
#[cold]
pub unsafe extern "C" fn dav1d_default_settings(s: NonNull<Dav1dSettings>) {
    let settings = Rav1dSettings::default().into();
    // SAFETY: `s` is safe to `ptr::write` to.
    unsafe { s.as_ptr().write(settings) };
}

/// # Safety
///
/// * `s`, if [`NonNull`], must valid to [`ptr::read`] from.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_get_frame_delay")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_get_frame_delay")]
#[cold]
pub unsafe extern "C" fn dav1d_get_frame_delay(s: Option<NonNull<Dav1dSettings>>) -> Dav1dResult {
    (|| {
        let s = validate_input!(s.ok_or(EINVAL))?;
        // SAFETY: `s` is safe to `ptr::read`.
        let s = unsafe { s.as_ptr().read() };
        let s = s.try_into()?;
        rav1d_get_frame_delay(&s).map(|frame_delay| frame_delay as c_uint)
    })()
    .into()
}

/// # Safety
///
/// * `c_out`, if [`NonNull`], is valid to [`ptr::write`] to.
/// * `s`, if [`NonNull`], is valid to [`ptr::read`] from.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_open")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_open")]
#[cold]
pub unsafe extern "C" fn dav1d_open(
    c_out: Option<NonNull<Option<Dav1dContext>>>,
    s: Option<NonNull<Dav1dSettings>>,
) -> Dav1dResult {
    (|| {
        let mut c_out = validate_input!(c_out.ok_or(EINVAL))?;
        let s = validate_input!(s.ok_or(EINVAL))?;
        // SAFETY: `c_out` is safe to write to.
        let c_out = unsafe { c_out.as_mut() };
        // SAFETY: `s` is safe to read from.
        let s = unsafe { s.as_ptr().read() };
        let s = s.try_into()?;
        let (c, handles) = rav1d_open(&s).inspect_err(|_| {
            *c_out = None;
        })?;

        // Spawn janitor thread to join worker handles on shutdown
        // C FFI can't properly manage Rust JoinHandles, so we spawn a helper thread
        if !handles.is_empty() {
            let ctx_clone = Arc::clone(&c);
            thread::spawn(move || {
                // Wait for die signal on all worker threads
                loop {
                    let all_died = ctx_clone.tc.iter().all(|tc| {
                        matches!(tc.task, Rav1dContextTaskType::Single(_))
                            || tc.thread_data.die.get()
                    });
                    if all_died {
                        break;
                    }
                    thread::sleep(Duration::from_millis(10));
                }

                // Join all worker threads
                for handle in handles {
                    let _ = handle.join();
                }
            });
        }

        *c_out = Some(RawArc::from_arc(c));
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `out`, if [`NonNull`], is valid to [`ptr::write`] to.
/// * `ptr`, if [`NonNull`], is the start of a `&[u8]` slice of length `sz`.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_parse_sequence_header")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_parse_sequence_header")]
pub unsafe extern "C" fn dav1d_parse_sequence_header(
    out: Option<NonNull<Dav1dSequenceHeader>>,
    ptr: Option<NonNull<u8>>,
    sz: usize,
) -> Dav1dResult {
    (|| {
        let out = validate_input!(out.ok_or(EINVAL))?;
        let ptr = validate_input!(ptr.ok_or(EINVAL))?;
        validate_input!((sz > 0 && sz <= usize::MAX / 2, EINVAL))?;
        // SAFETY: `ptr` is the start of a `&[u8]` slice of length `sz`.
        let data = unsafe { slice::from_raw_parts(ptr.as_ptr(), sz) };
        let seq_hdr = rav1d_parse_sequence_header(data)?.dav1d;
        // SAFETY: `out` is safe to write to.
        unsafe { out.as_ptr().write(seq_hdr) };
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `r#in`, if [`NonNull`], must be valid to [`ptr::read`] from and [`ptr::write`] to.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_send_data")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_send_data")]
pub unsafe extern "C" fn dav1d_send_data(
    c: Option<Dav1dContext>,
    r#in: Option<NonNull<Dav1dData>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(EINVAL))?;
        let r#in = validate_input!(r#in.ok_or(EINVAL))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        // SAFETY: `r#in` is safe to read from.
        let in_c = unsafe { r#in.as_ptr().read() };
        let mut in_rust = in_c.into();
        let result = rav1d_send_data(c, &mut in_rust);
        let in_c = in_rust.into();
        // SAFETY: `r#in` is safe to write to.
        unsafe { r#in.as_ptr().write(in_c) };
        result
    })()
    .into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `out`, if [`NonNull`], must be valid to [`ptr::write`] to.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_get_picture")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_get_picture")]
pub unsafe extern "C" fn dav1d_get_picture(
    c: Option<Dav1dContext>,
    out: Option<NonNull<Dav1dPicture>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(EINVAL))?;
        let out = validate_input!(out.ok_or(EINVAL))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        let mut out_rust = Default::default(); // TODO(kkysen) Temporary until we return it directly.
        let result = rav1d_get_picture(c, &mut out_rust);
        let out_c = out_rust.into();
        // SAFETY: `out` is safe to write to.
        unsafe { out.as_ptr().write(out_c) };
        result
    })()
    .into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `out`, if [`NonNull`], must be valid to [`ptr::write`] to.
/// * `r#in`, if [`NonNull`], must be valid to [`ptr::read`] from.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_apply_grain")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_apply_grain")]
pub unsafe extern "C" fn dav1d_apply_grain(
    c: Option<Dav1dContext>,
    out: Option<NonNull<Dav1dPicture>>,
    r#in: Option<NonNull<Dav1dPicture>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(EINVAL))?;
        let out = validate_input!(out.ok_or(EINVAL))?;
        let r#in = validate_input!(r#in.ok_or(EINVAL))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        // SAFETY: `r#in` is safe to read from.
        let in_c = unsafe { r#in.as_ptr().read() };
        // Don't `.update_rav1d()` [`Rav1dSequenceHeader`] because it's meant to be read-only.
        // Don't `.update_rav1d()` [`Rav1dFrameHeader`] because it's meant to be read-only.
        // Don't `.update_rav1d()` [`Rav1dITUTT35`] because we never read it.
        let mut out_rust = Default::default(); // TODO(kkysen) Temporary until we return it directly.
        let in_rust = in_c.into();
        let result = rav1d_apply_grain(c, &mut out_rust, &in_rust);
        let out_c = out_rust.into();
        // SAFETY: `out` is safe to write to.
        unsafe { out.as_ptr().write(out_c) };
        result
    })()
    .into()
}

/// # Safety
///
/// * `c` must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_flush")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_flush")]
pub unsafe extern "C" fn dav1d_flush(c: Dav1dContext) {
    // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
    // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
    let c = unsafe { c.as_ref() };
    rav1d_flush(c)
}

/// # Safety
///
/// * `c_out`, if [`NonNull`], must be safe to [`ptr::read`] from and [`ptr::write`] to.
///   The `Dav1dContext` pointed to by `c_out` must be from [`dav1d_open`].
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_close")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_close")]
#[cold]
pub unsafe extern "C" fn dav1d_close(c_out: Option<NonNull<Option<Dav1dContext>>>) {
    let Ok(mut c_out) = validate_input!(c_out.ok_or(())) else {
        return;
    };
    // SAFETY: `c_out` is safe to read from and write to.
    let c_out = unsafe { c_out.as_mut() };
    mem::take(c_out).map(|c| {
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        let c = unsafe { c.into_arc() };
        rav1d_close(c);
    });
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `flags`, if [`NonNull`], must be valid to [`ptr::write`] to.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_get_event_flags")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_get_event_flags")]
pub unsafe extern "C" fn dav1d_get_event_flags(
    c: Option<Dav1dContext>,
    flags: Option<NonNull<Dav1dEventFlags>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(EINVAL))?;
        let flags = validate_input!(flags.ok_or(EINVAL))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        let state = &mut *c.state.try_lock().unwrap();
        let flags_rust = mem::take(&mut state.event_flags);
        let flags_c = flags_rust.into();
        // SAFETY: `flags` is safe to write to.
        unsafe { flags.as_ptr().write(flags_c) };
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `c`, if [`NonNull`], must be from [`dav1d_open`] and not be passed to [`dav1d_close`] yet.
/// * `out`, if [`NonNull`], is valid to [`ptr::write`] to.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_get_decode_error_data_props")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_get_decode_error_data_props")]
pub unsafe extern "C" fn dav1d_get_decode_error_data_props(
    c: Option<Dav1dContext>,
    out: Option<NonNull<Dav1dDataProps>>,
) -> Dav1dResult {
    (|| {
        let c = validate_input!(c.ok_or(EINVAL))?;
        let out = validate_input!(out.ok_or(EINVAL))?;
        // SAFETY: `c` is from `dav1d_open` and thus from `RawArc::from_arc`.
        // It has not yet been passed to `dav1d_close` and thus not to `RawArc::into_arc` yet.
        let c = unsafe { c.as_ref() };
        let state = &mut *c.state.try_lock().unwrap();
        let props_rust = mem::take(&mut state.cached_error_props);
        let props_c = props_rust.into();
        // SAFETY: `out` is safety to write to.
        unsafe { out.as_ptr().write(props_c) };
        Ok(())
    })()
    .into()
}

/// # Safety
///
/// * `p`, if [`NonNull`], must be valid to [`ptr::read`] from and [`ptr::write`] to.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_picture_unref")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_picture_unref")]
pub unsafe extern "C" fn dav1d_picture_unref(p: Option<NonNull<Dav1dPicture>>) {
    let Ok(p) = validate_input!(p.ok_or(())) else {
        return;
    };
    // SAFETY: `p` is safe to read from.
    let p_c = unsafe { p.as_ptr().read() };
    let mut p_rust = p_c.to::<Rav1dPicture>();
    let _ = mem::take(&mut p_rust);
    let p_c = p_rust.into();
    // SAFETY: `p` is safe to write to.
    unsafe { p.as_ptr().write(p_c) };
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is valid to [`ptr::write`] to.
///   After this call, `buf.data` will be an allocated slice of length `sz`.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_data_create")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_data_create")]
pub unsafe extern "C" fn dav1d_data_create(buf: Option<NonNull<Dav1dData>>, sz: usize) -> *mut u8 {
    || -> Rav1dResult<*mut u8> {
        let buf = validate_input!(buf.ok_or(EINVAL))?;
        validate_input!((sz <= usize::MAX / 2, EINVAL))?;
        let data = Rav1dData::create(sz)?;
        let data = data.to::<Dav1dData>();
        let ptr = data
            .data
            .map(|ptr| ptr.as_ptr())
            .unwrap_or_else(ptr::null_mut);
        // SAFETY: `buf` is safe to write to.
        unsafe { buf.as_ptr().write(data) };
        Ok(ptr)
    }()
    .unwrap_or_else(|_| ptr::null_mut())
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is valid to [`ptr::write`] to.
/// * `ptr`, if [`NonNull`], is the start of a `&[u8]` slice of length `sz`.
/// * `ptr`'s slice must be valid to dereference until `free_callback` is called on it, which must deallocate it.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_data_wrap")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_data_wrap")]
pub unsafe extern "C" fn dav1d_data_wrap(
    buf: Option<NonNull<Dav1dData>>,
    ptr: Option<NonNull<u8>>,
    sz: usize,
    free_callback: Option<FnFree>,
    user_data: Option<SendSyncNonNull<c_void>>,
) -> Dav1dResult {
    || -> Rav1dResult {
        let buf = validate_input!(buf.ok_or(EINVAL))?;
        let ptr = validate_input!(ptr.ok_or(EINVAL))?;
        validate_input!((sz <= usize::MAX / 2, EINVAL))?;
        // SAFETY: `ptr` is the start of a `&[u8]` slice of length `sz`.
        let data = unsafe { slice::from_raw_parts(ptr.as_ptr(), sz) };
        // SAFETY: `ptr`, and thus `data`, is valid to dereference until `free_callback` is called on it, which deallocates it.
        let data = unsafe { Rav1dData::wrap(data.into(), free_callback, user_data) }?;
        let data_c = data.into();
        // SAFETY: `buf` is safe to write to.
        unsafe { buf.as_ptr().write(data_c) };
        Ok(())
    }()
    .into()
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is valid to [`ptr::read`] from and [`ptr::write`] to.
/// * `user_data`, if [`NonNull`], is valid to dereference until `free_callback` is called on it, which must deallocate it.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_data_wrap_user_data")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_data_wrap_user_data")]
pub unsafe extern "C" fn dav1d_data_wrap_user_data(
    buf: Option<NonNull<Dav1dData>>,
    user_data: Option<NonNull<u8>>,
    free_callback: Option<FnFree>,
    cookie: Option<SendSyncNonNull<c_void>>,
) -> Dav1dResult {
    || -> Rav1dResult {
        let buf = validate_input!(buf.ok_or(EINVAL))?;
        // Note that `dav1d` doesn't do this check, but they do for the similar [`dav1d_data_wrap`].
        let user_data = validate_input!(user_data.ok_or(EINVAL))?;
        // SAFETY: `buf` is safe to read from.
        let data_c = unsafe { buf.as_ptr().read() };
        let mut data = data_c.to::<Rav1dData>();
        // SAFETY: `user_data` is valid to dereference until `free_callback` is called on it, which deallocates it.
        unsafe { data.wrap_user_data(user_data, free_callback, cookie) }?;
        let data_c = data.into();
        // SAFETY: `buf` is safe to write to.
        unsafe { buf.as_ptr().write(data_c) };
        Ok(())
    }()
    .into()
}

/// # Safety
///
/// * `buf`, if [`NonNull`], is safe to [`ptr::read`] from and [`ptr::write`] from.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_data_unref")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_data_unref")]
pub unsafe extern "C" fn dav1d_data_unref(buf: Option<NonNull<Dav1dData>>) {
    let buf = validate_input!(buf.ok_or(()));
    let Ok(mut buf) = buf else { return };
    // SAFETY: `buf` is safe to read from and write to.
    let buf = unsafe { buf.as_mut() };
    let _ = mem::take(buf).to::<Rav1dData>();
}

/// # Safety
///
/// * `props`, if [`NonNull`], is safe to [`ptr::read`] from and [`ptr::write`] from.
#[cfg_attr(feature = "dav1d-compat", export_name = "dav1d_data_props_unref")]
#[cfg_attr(not(feature = "dav1d-compat"), export_name = "rav1d_data_props_unref")]
pub unsafe extern "C" fn dav1d_data_props_unref(props: Option<NonNull<Dav1dDataProps>>) {
    let props = validate_input!(props.ok_or(()));
    let Ok(mut props) = props else { return };
    // SAFETY: `props` is safe to read from and write to.
    let props = unsafe { props.as_mut() };
    let _ = mem::take(props).to::<Rav1dDataProps>();
}
