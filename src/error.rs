#![forbid(unsafe_code)]
use std::ffi::c_int;
use std::ffi::c_uint;
use strum::FromRepr;

#[derive(Clone, Copy, PartialEq, Eq, FromRepr, Debug)]
#[repr(u8)]
#[non_exhaustive]
pub enum Rav1dError {
    /// This represents a generic `rav1d` error.
    /// It has nothing to do with the other `errno`-based ones
    /// (and that's why it's not all caps like the other ones).
    ///
    /// Normally `EPERM = 1`, but `dav1d` never uses `EPERM`,
    /// but does use `-1`, as opposed to the normal `DAV1D_ERR(E*)`.
    ///
    /// Also Note that this forces `0` to be the niche,
    /// which is more optimal since `0` is no error for [`Dav1dResult`].
    EGeneric = 1,

    // POSIX errno values as dav1d's Linux ABI numbers them. These are the
    // crate's stable Rust-side identities; the C boundary maps each variant to
    // the PLATFORM's errno via `errno()` / `from_errno()` (macOS `EAGAIN` is
    // 35, Windows `ENOPROTOOPT` is 109, …), so a C caller comparing against its
    // own `<errno.h>` sees the value dav1d would have returned on that platform.
    ENOENT = 2,
    EIO = 5,
    EAGAIN = 11,
    ENOMEM = 12,
    EINVAL = 22,
    ERANGE = 34,
    ENOPROTOOPT = 92,

    /// Decode was stopped cooperatively via a [`Stop`](enough::Stop) token
    /// (issue #412). Not a POSIX passthrough — this is an internal rav1d signal,
    /// so it is intentionally excluded from the `c-ffi` errno static assertions
    /// below. The numeric value matches Linux `ECANCELED` (125) for the common
    /// case, but the managed API authoritatively reports cancellation by
    /// re-checking the caller's token, not by this code.
    ECANCELED = 125,
}

impl Rav1dError {
    /// The platform errno a C caller expects for this error (positive; the
    /// `DAV1D_ERR` negation happens at the [`Dav1dResult`] boundary).
    ///
    /// With `c-ffi` this consults `libc`, so the value is correct on macOS,
    /// Windows and the BSDs, not just Linux. Without `c-ffi` the discriminant
    /// (dav1d's Linux numbering) is returned unchanged.
    #[inline]
    pub const fn errno(self) -> c_int {
        #[cfg(feature = "c-ffi")]
        {
            match self {
                Self::EGeneric => 1,
                Self::ENOENT => libc::ENOENT,
                Self::EIO => libc::EIO,
                Self::EAGAIN => libc::EAGAIN,
                Self::ENOMEM => libc::ENOMEM,
                Self::EINVAL => libc::EINVAL,
                Self::ERANGE => libc::ERANGE,
                Self::ENOPROTOOPT => libc::ENOPROTOOPT,
                Self::ECANCELED => libc::ECANCELED,
            }
        }
        #[cfg(not(feature = "c-ffi"))]
        {
            self as c_int
        }
    }

    /// Inverse of [`errno`](Self::errno): a positive platform errno back to the
    /// variant, `None` for anything dav1d never returns.
    #[inline]
    pub const fn from_errno(errno: c_int) -> Option<Self> {
        #[cfg(feature = "c-ffi")]
        {
            // `match` on non-literal consts is not allowed; a chain keeps this
            // `const fn` and platform-correct.
            if errno == 1 {
                Some(Self::EGeneric)
            } else if errno == libc::ENOENT {
                Some(Self::ENOENT)
            } else if errno == libc::EIO {
                Some(Self::EIO)
            } else if errno == libc::EAGAIN {
                Some(Self::EAGAIN)
            } else if errno == libc::ENOMEM {
                Some(Self::ENOMEM)
            } else if errno == libc::EINVAL {
                Some(Self::EINVAL)
            } else if errno == libc::ERANGE {
                Some(Self::ERANGE)
            } else if errno == libc::ENOPROTOOPT {
                Some(Self::ENOPROTOOPT)
            } else if errno == libc::ECANCELED {
                Some(Self::ECANCELED)
            } else {
                None
            }
        }
        #[cfg(not(feature = "c-ffi"))]
        {
            if errno < 0 || errno > u8::MAX as c_int {
                return None;
            }
            Self::from_repr(errno as u8)
        }
    }
}

// The discriminants ARE the Linux errno values (dav1d's reference ABI). Pin
// that where libc can confirm it; other platforms go through `errno()`.
#[cfg(all(feature = "c-ffi", target_os = "linux"))]
const _: () = {
    assert!(Rav1dError::ENOENT as c_int == libc::ENOENT);
    assert!(Rav1dError::EIO as c_int == libc::EIO);
    assert!(Rav1dError::EAGAIN as c_int == libc::EAGAIN);
    assert!(Rav1dError::ENOMEM as c_int == libc::ENOMEM);
    assert!(Rav1dError::EINVAL as c_int == libc::EINVAL);
    assert!(Rav1dError::ERANGE as c_int == libc::ERANGE);
    assert!(Rav1dError::ENOPROTOOPT as c_int == libc::ENOPROTOOPT);
    assert!(Rav1dError::ECANCELED as c_int == libc::ECANCELED);
};

pub type Rav1dResult<T = ()> = Result<T, Rav1dError>;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Dav1dResult(pub c_int);

impl From<Rav1dResult> for Dav1dResult {
    #[inline]
    fn from(value: Rav1dResult) -> Self {
        // Doing the `-` negation on both branches
        // makes the code short and branchless.
        Dav1dResult(
            -(match value {
                Ok(()) => 0,
                Err(e) => e.errno(),
            }),
        )
    }
}

impl From<Rav1dResult<c_uint>> for Dav1dResult {
    #[inline]
    fn from(value: Rav1dResult<c_uint>) -> Self {
        Dav1dResult(match value {
            Ok(value) => value as c_int,
            Err(e) => e.errno(),
        })
    }
}

impl TryFrom<Dav1dResult> for Rav1dResult {
    type Error = Dav1dResult;

    #[inline]
    fn try_from(value: Dav1dResult) -> Result<Self, Self::Error> {
        match value.0 {
            0 => Ok(Ok(())),
            e => {
                let e = Rav1dError::from_errno(-e).ok_or(value)?;
                Ok(Err(e))
            }
        }
    }
}

#[cfg(test)]
mod errno_tests {
    use super::*;

    const ALL: [Rav1dError; 9] = [
        Rav1dError::EGeneric,
        Rav1dError::ENOENT,
        Rav1dError::EIO,
        Rav1dError::EAGAIN,
        Rav1dError::ENOMEM,
        Rav1dError::EINVAL,
        Rav1dError::ERANGE,
        Rav1dError::ENOPROTOOPT,
        Rav1dError::ECANCELED,
    ];

    /// The C boundary must round-trip every variant on THIS platform.
    #[test]
    fn errno_round_trips_on_this_platform() {
        for e in ALL {
            assert!(e.errno() > 0, "{e:?} must map to a positive errno");
            assert_eq!(Rav1dError::from_errno(e.errno()), Some(e), "{e:?}");
            assert_eq!(
                Rav1dResult::try_from(Dav1dResult::from(Err::<(), _>(e))),
                Ok(Err(e)),
                "{e:?} through Dav1dResult"
            );
        }
        assert_eq!(Rav1dResult::try_from(Dav1dResult(0)), Ok(Ok(())));
        assert_eq!(Rav1dError::from_errno(0), None);
        assert_eq!(Rav1dError::from_errno(-1), None);
    }

    /// With `c-ffi` the boundary speaks the platform's errno, which is what a
    /// C caller compares against (macOS EAGAIN is 35, Linux 11).
    #[cfg(feature = "c-ffi")]
    #[test]
    fn errno_is_the_platform_value_under_c_ffi() {
        assert_eq!(Rav1dError::EAGAIN.errno(), libc::EAGAIN);
        assert_eq!(Rav1dError::ENOPROTOOPT.errno(), libc::ENOPROTOOPT);
        assert_eq!(Rav1dError::ECANCELED.errno(), libc::ECANCELED);
        assert_eq!(
            Dav1dResult::from(Err::<(), _>(Rav1dError::EAGAIN)).0,
            -libc::EAGAIN
        );
    }
}
