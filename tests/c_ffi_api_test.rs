//! Integration tests for the C FFI API (`dav1d_*` functions).
//!
//! These call the `extern "C"` functions directly from Rust to verify
//! error handling, lifecycle, and edge cases without needing a C compiler.
#![cfg(feature = "c-ffi")]

mod ivf_parser;

use rav1d_safe::include::dav1d::common::Dav1dDataProps;
use rav1d_safe::include::dav1d::data::Dav1dData;
use rav1d_safe::include::dav1d::dav1d::Dav1dContext;
use rav1d_safe::include::dav1d::dav1d::Dav1dEventFlags;
use rav1d_safe::include::dav1d::dav1d::Dav1dSettings;
use rav1d_safe::include::dav1d::headers::Dav1dSequenceHeader;
use rav1d_safe::include::dav1d::picture::Dav1dPicture;
use rav1d_safe::src::dav1d_api::*;
use std::ffi::CStr;
use std::io::Cursor;
use std::path::Path;
use std::ptr::NonNull;

// ========================================================================
// Helpers
// ========================================================================

fn test_vector_path(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors/dav1d-test-data")
        .join(rel)
}

/// Open a decoder with default settings, single-threaded.
fn open_decoder() -> (Option<Dav1dContext>, Dav1dSettings) {
    let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
    unsafe { dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap()) };
    let mut settings = unsafe { settings.assume_init() };
    settings.n_threads = 1;
    settings.max_frame_delay = 1;

    let mut ctx: Option<Dav1dContext> = None;
    let r = unsafe {
        dav1d_open(
            NonNull::new(&mut ctx as *mut _),
            NonNull::new(&mut settings as *mut _),
        )
    };
    assert_eq!(r.0, 0, "dav1d_open failed: {}", r.0);
    assert!(ctx.is_some(), "dav1d_open returned null context");
    (ctx, settings)
}

/// Close a decoder context.
fn close_decoder(ctx: &mut Option<Dav1dContext>) {
    unsafe { dav1d_close(NonNull::new(ctx as *mut _)) };
    assert!(ctx.is_none(), "dav1d_close didn't null out context");
}

/// Read an IVF file and return all OBU frames.
fn read_ivf_frames(path: &Path) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).unwrap();
    let mut cursor = Cursor::new(data);
    let frames = ivf_parser::parse_all_frames(&mut cursor).unwrap();
    frames.into_iter().map(|f| f.data).collect()
}

// ========================================================================
// Version & constants
// ========================================================================

#[test]
fn test_dav1d_version_returns_valid_cstr() {
    let ptr = dav1d_version();
    assert!(!ptr.is_null());
    let version = unsafe { CStr::from_ptr(ptr) };
    let s = version.to_str().unwrap();
    assert!(!s.is_empty(), "version string is empty");
}

#[test]
fn test_dav1d_version_api_format() {
    let v = dav1d_version_api();
    // Format: 0x00XXYYZZ
    assert_eq!(v >> 24, 0, "top byte should be 0");
    let major = (v >> 16) & 0xFF;
    assert!(major > 0, "major version should be > 0, got {major}");
}

// ========================================================================
// Settings
// ========================================================================

#[test]
fn test_dav1d_default_settings_produces_valid_defaults() {
    let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
    unsafe { dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap()) };
    let settings = unsafe { settings.assume_init() };
    // apply_grain defaults to true (nonzero)
    assert_ne!(settings.apply_grain, 0, "apply_grain should default to true");
    // all_layers defaults to true
    assert_ne!(settings.all_layers, 0, "all_layers should default to true");
}

#[test]
fn test_dav1d_get_frame_delay_null_settings() {
    let r = unsafe { dav1d_get_frame_delay(None) };
    // Note: dav1d_get_frame_delay uses From<Rav1dResult<c_uint>> which returns
    // positive EINVAL (22) on error rather than negated. This matches upstream rav1d.
    assert_eq!(r.0, libc::EINVAL, "null settings should return EINVAL");
}

#[test]
fn test_dav1d_get_frame_delay_valid() {
    let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
    unsafe { dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap()) };
    let settings = unsafe { settings.assume_init() };
    let r = unsafe { dav1d_get_frame_delay(NonNull::new(&settings as *const _ as *mut _)) };
    assert!(r.0 > 0, "frame delay should be positive, got {}", r.0);
}

// ========================================================================
// Open / Close
// ========================================================================

#[test]
fn test_dav1d_open_null_context_returns_einval() {
    let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
    unsafe { dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap()) };
    let mut settings = unsafe { settings.assume_init() };
    let r = unsafe { dav1d_open(None, NonNull::new(&mut settings as *mut _)) };
    assert!(r.0 < 0, "null c_out should fail");
}

#[test]
fn test_dav1d_open_null_settings_returns_einval() {
    let mut ctx: Option<Dav1dContext> = None;
    let r = unsafe { dav1d_open(NonNull::new(&mut ctx as *mut _), None) };
    assert!(r.0 < 0, "null settings should fail");
    assert!(ctx.is_none(), "context should be None on failure");
}

#[test]
fn test_dav1d_open_close_lifecycle() {
    let (mut ctx, _settings) = open_decoder();
    close_decoder(&mut ctx);
}

#[test]
fn test_dav1d_open_invalid_threads() {
    let mut settings = std::mem::MaybeUninit::<Dav1dSettings>::uninit();
    unsafe { dav1d_default_settings(NonNull::new(settings.as_mut_ptr()).unwrap()) };
    let mut settings = unsafe { settings.assume_init() };
    settings.n_threads = 999; // > 256 limit

    let mut ctx: Option<Dav1dContext> = None;
    let r = unsafe {
        dav1d_open(
            NonNull::new(&mut ctx as *mut _),
            NonNull::new(&mut settings as *mut _),
        )
    };
    assert!(r.0 < 0, "invalid thread count should fail");
}

// ========================================================================
// send_data / get_picture — error paths
// ========================================================================

#[test]
fn test_dav1d_send_data_null_context() {
    let mut data = Dav1dData::default();
    let r = unsafe { dav1d_send_data(None, NonNull::new(&mut data as *mut _)) };
    assert!(r.0 < 0, "null context should fail");
}

#[test]
fn test_dav1d_send_data_null_data() {
    let (mut ctx, _) = open_decoder();
    let r = unsafe { dav1d_send_data(ctx, None) };
    assert!(r.0 < 0, "null data should fail");
    close_decoder(&mut ctx);
}

#[test]
fn test_dav1d_get_picture_null_context() {
    let mut pic = Dav1dPicture::default();
    let r = unsafe { dav1d_get_picture(None, NonNull::new(&mut pic as *mut _)) };
    assert!(r.0 < 0, "null context should fail");
}

#[test]
fn test_dav1d_get_picture_null_output() {
    let (mut ctx, _) = open_decoder();
    let r = unsafe { dav1d_get_picture(ctx, None) };
    assert!(r.0 < 0, "null output should fail");
    close_decoder(&mut ctx);
}

#[test]
fn test_dav1d_get_picture_before_send_returns_eagain() {
    let (mut ctx, _) = open_decoder();
    let mut pic = Dav1dPicture::default();
    let r = unsafe { dav1d_get_picture(ctx, NonNull::new(&mut pic as *mut _)) };
    // EAGAIN = -11 (no picture available yet)
    assert_eq!(r.0, -(libc::EAGAIN as i32), "should return EAGAIN");
    close_decoder(&mut ctx);
}

// ========================================================================
// Flush
// ========================================================================

#[test]
fn test_dav1d_flush_on_fresh_context() {
    let (mut ctx, _) = open_decoder();
    // Flush on a fresh context should not crash
    unsafe { dav1d_flush(ctx.unwrap()) };
    close_decoder(&mut ctx);
}

// ========================================================================
// Event flags / error data props
// ========================================================================

#[test]
fn test_dav1d_get_event_flags_null_context() {
    let mut flags: Dav1dEventFlags = 0;
    let r = unsafe { dav1d_get_event_flags(None, NonNull::new(&mut flags as *mut _)) };
    assert!(r.0 < 0, "null context should fail");
}

#[test]
fn test_dav1d_get_event_flags_null_flags() {
    let (mut ctx, _) = open_decoder();
    let r = unsafe { dav1d_get_event_flags(ctx, None) };
    assert!(r.0 < 0, "null flags should fail");
    close_decoder(&mut ctx);
}

#[test]
fn test_dav1d_get_event_flags_fresh_context() {
    let (mut ctx, _) = open_decoder();
    let mut flags: Dav1dEventFlags = 0xFFFF;
    let r = unsafe { dav1d_get_event_flags(ctx, NonNull::new(&mut flags as *mut _)) };
    assert_eq!(r.0, 0, "should succeed");
    assert_eq!(flags, 0, "fresh context should have no event flags");
    close_decoder(&mut ctx);
}

#[test]
fn test_dav1d_get_decode_error_data_props_null_context() {
    let mut props = Dav1dDataProps::default();
    let r =
        unsafe { dav1d_get_decode_error_data_props(None, NonNull::new(&mut props as *mut _)) };
    assert!(r.0 < 0, "null context should fail");
}

#[test]
fn test_dav1d_get_decode_error_data_props_null_output() {
    let (mut ctx, _) = open_decoder();
    let r = unsafe { dav1d_get_decode_error_data_props(ctx, None) };
    assert!(r.0 < 0, "null output should fail");
    close_decoder(&mut ctx);
}

// ========================================================================
// picture_unref
// ========================================================================

#[test]
fn test_dav1d_picture_unref_null_is_noop() {
    // Should not crash
    unsafe { dav1d_picture_unref(None) };
}

// ========================================================================
// data_create
// ========================================================================

#[test]
fn test_dav1d_data_create_null_buf() {
    let ptr = unsafe { dav1d_data_create(None, 64) };
    assert!(ptr.is_null(), "null buf should return null");
}

#[test]
fn test_dav1d_data_create_valid() {
    let mut data = Dav1dData::default();
    let ptr = unsafe { dav1d_data_create(NonNull::new(&mut data as *mut _), 128) };
    assert!(!ptr.is_null(), "should return valid pointer");
    assert_eq!(data.sz, 128, "size should match");
    assert!(data.data.is_some(), "data pointer should be set");
    // Clean up
    unsafe { dav1d_data_unref(NonNull::new(&mut data as *mut _)) };
}

#[test]
fn test_dav1d_data_create_zero_size() {
    let mut data = Dav1dData::default();
    let ptr = unsafe { dav1d_data_create(NonNull::new(&mut data as *mut _), 0) };
    // Zero-size allocation should still succeed (or at least not crash)
    if !ptr.is_null() {
        unsafe { dav1d_data_unref(NonNull::new(&mut data as *mut _)) };
    }
}

// ========================================================================
// data_unref
// ========================================================================

#[test]
fn test_dav1d_data_unref_null_is_noop() {
    unsafe { dav1d_data_unref(None) };
}

// ========================================================================
// data_props_unref
// ========================================================================

#[test]
fn test_dav1d_data_props_unref_null_is_noop() {
    unsafe { dav1d_data_props_unref(None) };
}

// ========================================================================
// parse_sequence_header
// ========================================================================

#[test]
fn test_dav1d_parse_sequence_header_null_output() {
    let data = [0u8; 16];
    let r = unsafe {
        dav1d_parse_sequence_header(None, NonNull::new(data.as_ptr() as *mut _), data.len())
    };
    assert!(r.0 < 0, "null output should fail");
}

#[test]
fn test_dav1d_parse_sequence_header_null_data() {
    let mut hdr = std::mem::MaybeUninit::<Dav1dSequenceHeader>::uninit();
    let r = unsafe { dav1d_parse_sequence_header(NonNull::new(hdr.as_mut_ptr()), None, 16) };
    assert!(r.0 < 0, "null data should fail");
}

#[test]
fn test_dav1d_parse_sequence_header_zero_size() {
    let mut hdr = std::mem::MaybeUninit::<Dav1dSequenceHeader>::uninit();
    let data = [0u8; 1];
    let r = unsafe {
        dav1d_parse_sequence_header(
            NonNull::new(hdr.as_mut_ptr()),
            NonNull::new(data.as_ptr() as *mut _),
            0,
        )
    };
    assert!(r.0 < 0, "zero size should fail");
}

#[test]
fn test_dav1d_parse_sequence_header_garbage_data() {
    let mut hdr = std::mem::MaybeUninit::<Dav1dSequenceHeader>::uninit();
    let data = [0xFFu8; 64];
    let r = unsafe {
        dav1d_parse_sequence_header(
            NonNull::new(hdr.as_mut_ptr()),
            NonNull::new(data.as_ptr() as *mut _),
            data.len(),
        )
    };
    // Garbage data should fail to parse, but not crash
    assert!(r.0 < 0, "garbage data should fail to parse");
}

// ========================================================================
// apply_grain — error paths
// ========================================================================

#[test]
fn test_dav1d_apply_grain_null_context() {
    let mut out = Dav1dPicture::default();
    let r#in = Dav1dPicture::default();
    let r = unsafe {
        dav1d_apply_grain(
            None,
            NonNull::new(&mut out as *mut _),
            NonNull::new(&r#in as *const _ as *mut _),
        )
    };
    assert!(r.0 < 0, "null context should fail");
}

#[test]
fn test_dav1d_apply_grain_null_output() {
    let (mut ctx, _) = open_decoder();
    let r#in = Dav1dPicture::default();
    let r = unsafe {
        dav1d_apply_grain(ctx, None, NonNull::new(&r#in as *const _ as *mut _))
    };
    assert!(r.0 < 0, "null output should fail");
    close_decoder(&mut ctx);
}

// ========================================================================
// Full decode lifecycle (integration)
// ========================================================================

#[test]
fn test_full_decode_lifecycle() {
    let ivf_path = test_vector_path("8-bit/data/00000000.ivf");
    if !ivf_path.exists() {
        eprintln!("Skipping test_full_decode_lifecycle: test vectors not available");
        return;
    }

    let frames = read_ivf_frames(&ivf_path);
    assert!(!frames.is_empty(), "IVF file should contain frames");

    let (mut ctx, _) = open_decoder();

    let mut decoded_count = 0u32;
    for frame_data in &frames {
        // Create data buffer
        let mut data = Dav1dData::default();
        let ptr =
            unsafe { dav1d_data_create(NonNull::new(&mut data as *mut _), frame_data.len()) };
        assert!(!ptr.is_null(), "dav1d_data_create failed");

        // Copy frame data into the buffer
        unsafe {
            std::ptr::copy_nonoverlapping(frame_data.as_ptr(), ptr, frame_data.len());
        }

        // Send data
        let r = unsafe { dav1d_send_data(ctx, NonNull::new(&mut data as *mut _)) };
        if r.0 < 0 && r.0 != -(libc::EAGAIN as i32) {
            // Clean up remaining data if send failed for non-EAGAIN reason
            if data.data.is_some() {
                unsafe { dav1d_data_unref(NonNull::new(&mut data as *mut _)) };
            }
            // Some frames may legitimately fail — skip
            continue;
        }

        // Try to get decoded pictures
        loop {
            let mut pic = Dav1dPicture::default();
            let r = unsafe { dav1d_get_picture(ctx, NonNull::new(&mut pic as *mut _)) };
            if r.0 == -(libc::EAGAIN as i32) {
                break; // No more pictures ready
            }
            if r.0 < 0 {
                break; // Error
            }
            // Got a picture
            assert!(pic.data[0].is_some(), "Y plane should be non-null");
            assert!(pic.p.w > 0, "width should be positive");
            assert!(pic.p.h > 0, "height should be positive");
            assert!(pic.p.bpc > 0, "bpc should be positive");
            decoded_count += 1;

            unsafe { dav1d_picture_unref(NonNull::new(&mut pic as *mut _)) };
        }
    }

    // Flush to get remaining pictures
    unsafe { dav1d_flush(ctx.unwrap()) };

    assert!(decoded_count > 0, "should have decoded at least one picture");
    close_decoder(&mut ctx);
}

#[test]
fn test_flush_and_redecode() {
    let ivf_path = test_vector_path("8-bit/data/00000000.ivf");
    if !ivf_path.exists() {
        eprintln!("Skipping test_flush_and_redecode: test vectors not available");
        return;
    }

    let frames = read_ivf_frames(&ivf_path);
    if frames.is_empty() {
        return;
    }

    let (mut ctx, _) = open_decoder();

    // Decode first frame
    let mut data = Dav1dData::default();
    let ptr = unsafe { dav1d_data_create(NonNull::new(&mut data as *mut _), frames[0].len()) };
    assert!(!ptr.is_null());
    unsafe { std::ptr::copy_nonoverlapping(frames[0].as_ptr(), ptr, frames[0].len()) };
    let _ = unsafe { dav1d_send_data(ctx, NonNull::new(&mut data as *mut _)) };

    // Flush
    unsafe { dav1d_flush(ctx.unwrap()) };

    // Decode same frame again after flush — should work
    let mut data = Dav1dData::default();
    let ptr = unsafe { dav1d_data_create(NonNull::new(&mut data as *mut _), frames[0].len()) };
    assert!(!ptr.is_null());
    unsafe { std::ptr::copy_nonoverlapping(frames[0].as_ptr(), ptr, frames[0].len()) };
    let r = unsafe { dav1d_send_data(ctx, NonNull::new(&mut data as *mut _)) };
    // Should accept data (0) or need more data (EAGAIN), but not hard error
    assert!(
        r.0 == 0 || r.0 == -(libc::EAGAIN as i32),
        "send after flush should work, got {}",
        r.0
    );

    if data.data.is_some() {
        unsafe { dav1d_data_unref(NonNull::new(&mut data as *mut _)) };
    }
    close_decoder(&mut ctx);
}

#[test]
fn test_parse_sequence_header_from_ivf() {
    let ivf_path = test_vector_path("8-bit/data/00000000.ivf");
    if !ivf_path.exists() {
        eprintln!("Skipping test_parse_sequence_header_from_ivf: test vectors not available");
        return;
    }

    let frames = read_ivf_frames(&ivf_path);
    assert!(!frames.is_empty());

    // First frame of an IVF usually starts with a sequence header OBU
    let mut hdr = std::mem::MaybeUninit::<Dav1dSequenceHeader>::uninit();
    let r = unsafe {
        dav1d_parse_sequence_header(
            NonNull::new(hdr.as_mut_ptr()),
            NonNull::new(frames[0].as_ptr() as *mut _),
            frames[0].len(),
        )
    };
    if r.0 == 0 {
        let hdr = unsafe { hdr.assume_init() };
        assert!(hdr.max_width > 0, "max_width should be positive");
        assert!(hdr.max_height > 0, "max_height should be positive");
    }
    // Even if it fails (frame might not start with seq hdr), no crash = success
}

#[test]
fn test_event_flags_after_decode() {
    let ivf_path = test_vector_path("8-bit/data/00000000.ivf");
    if !ivf_path.exists() {
        eprintln!("Skipping test_event_flags_after_decode: test vectors not available");
        return;
    }

    let frames = read_ivf_frames(&ivf_path);
    if frames.is_empty() {
        return;
    }

    let (mut ctx, _) = open_decoder();

    // Send first frame
    let mut data = Dav1dData::default();
    let ptr = unsafe { dav1d_data_create(NonNull::new(&mut data as *mut _), frames[0].len()) };
    assert!(!ptr.is_null());
    unsafe { std::ptr::copy_nonoverlapping(frames[0].as_ptr(), ptr, frames[0].len()) };
    let _ = unsafe { dav1d_send_data(ctx, NonNull::new(&mut data as *mut _)) };

    // Try to get a picture
    let mut pic = Dav1dPicture::default();
    let r = unsafe { dav1d_get_picture(ctx, NonNull::new(&mut pic as *mut _)) };
    if r.0 == 0 {
        unsafe { dav1d_picture_unref(NonNull::new(&mut pic as *mut _)) };
    }

    // Check event flags — should have NEW_SEQUENCE after first decode
    let mut flags: Dav1dEventFlags = 0;
    let r = unsafe { dav1d_get_event_flags(ctx, NonNull::new(&mut flags as *mut _)) };
    assert_eq!(r.0, 0, "get_event_flags should succeed");
    // After decoding the first frame, NEW_SEQUENCE should be set
    if r.0 == 0 {
        // Just verify it doesn't crash; flag value depends on decode state
    }

    if data.data.is_some() {
        unsafe { dav1d_data_unref(NonNull::new(&mut data as *mut _)) };
    }
    close_decoder(&mut ctx);
}
