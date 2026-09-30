#!/usr/bin/env bash
# Download + extract the Argon AV1 conformance suite.
#
# Usage: download-argon.sh [target_dir]
#
#   [target_dir]  Parent dir for the extracted suite (default:
#                 <repo>/test-vectors/argon). The suite lands in a versioned
#                 subdir (argon_coveragetool_.../ or argon/).
#
# Environment:
#   ARGON_URL   Archive URL. Default: the official AOM CWG S3 release
#               (v2.1.1 zip). Point at an internal mirror (e.g. R2) to avoid
#               the public origin. Format is sniffed from magic bytes, so the
#               mirror can repack as .zip or .tar.zst without breaking this.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TARGET="${1:-$PROJECT_ROOT/test-vectors/argon}"
ARGON_URL="${ARGON_URL:-https://aom-cwg-av1-argon-streams-public.s3.us-east-1.amazonaws.com/argon_coveragetool_av1_base_and_extended_profiles_v2.1.1.zip}"

mkdir -p "$TARGET"

# Already extracted? (current v2.x root, or the older streams.videolan.org
# argon/ layout used by earlier revisions of this script)
if compgen -G "$TARGET/argon_coveragetool_*" > /dev/null || [ -d "$TARGET/argon" ]; then
    echo "✓ Argon suite already extracted"
    exit 0
fi

ARCHIVE="$TARGET/argon_archive"
echo "→ Downloading Argon suite from $ARGON_URL"
wget -q --show-progress "$ARGON_URL" -O "$ARCHIVE"
ls -lh "$ARCHIVE"

echo "→ Extracting..."
cd "$TARGET"
magic="$(head -c4 "$ARCHIVE" | xxd -p)"
case "$magic" in
    504b0304)
        # Extract only what argon_md5_check.sh reads: streams + md5 sidecars
        # (+ the list manifests). The ArgonViewer HTML tool, coverage-tool
        # binaries, and per-stream ref_cmd/*.sh don't match these patterns.
        # *_error streams carry no sidecars and are designed to fail decode.
        unzip -q -o "$ARCHIVE" \
            '*/streams/*.obu' '*/md5_ref/*' '*/md5_no_film_grain/*' \
            '*/all_list.txt' '*/level*_list.txt' \
            -x '*_error/*'
        ;;
    fd2fb528)
        tar --use-compress-program=unzstd -xf "$ARCHIVE"
        ;;
    *)
        echo "Unknown archive magic: $magic" >&2
        exit 1
        ;;
esac
rm -f "$ARCHIVE"
echo "✓ Argon suite extracted to $TARGET"
