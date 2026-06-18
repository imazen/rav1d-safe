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

    // POSIX errno values. When c-ffi is enabled, a static assertion
    // verifies these match the platform's libc constants.
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

// When c-ffi is enabled, verify our hardcoded values match the platform's errno.
#[cfg(feature = "c-ffi")]
const _: () = {
    assert!(Rav1dError::ENOENT as u8 == libc::ENOENT as u8);
    assert!(Rav1dError::EIO as u8 == libc::EIO as u8);
    assert!(Rav1dError::EAGAIN as u8 == libc::EAGAIN as u8);
    assert!(Rav1dError::ENOMEM as u8 == libc::ENOMEM as u8);
    assert!(Rav1dError::EINVAL as u8 == libc::EINVAL as u8);
    assert!(Rav1dError::ERANGE as u8 == libc::ERANGE as u8);
    assert!(Rav1dError::ENOPROTOOPT as u8 == libc::ENOPROTOOPT as u8);
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
                Err(e) => e as c_int,
            }),
        )
    }
}

impl From<Rav1dResult<c_uint>> for Dav1dResult {
    #[inline]
    fn from(value: Rav1dResult<c_uint>) -> Self {
        Dav1dResult(match value {
            Ok(value) => value as c_int,
            Err(e) => e as c_int,
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
                let e = (-e).try_into().map_err(|_| value)?;
                let e = Rav1dError::from_repr(e).ok_or(value)?;
                Ok(Err(e))
            }
        }
    }
}
