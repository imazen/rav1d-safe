#!/usr/bin/env bash
# Build the CONTENT-CLASS decode corpus: photo vs screen-text vs screen-UI, at
# one pixel count, across a quality ladder.
#
# WHY THIS EXISTS
# ---------------
# Every perf number in the rav1d-safe campaign up to 2026-08-10 was taken on
# ONE content class (downscales of one photograph, `mk_size_ladder.sh`). That
# turned out to be the wrong class: at a single pixel count the ours/dav1d ratio
# spans 1.4x-3.2x depending only on CONTENT, which is wider than the whole size
# ladder's spread. Screen content pays a per-block tracker tax that photo does
# not, because it codes far more, far smaller blocks per pixel.
#
# So this corpus exists to make "does the change help the class that misses the
# bar" answerable. It is deliberately NOT a size ladder: pixels are held
# constant so content is the only free variable.
#
# WHAT IT MAKES
#   photo   — the campaign's own 4K photo (test-vectors/bench/photo_4k.avif),
#             decoded and downscaled. The CONTROL: the class every prior number
#             was taken on.
#   text    — a Quick Look render of two of this repo's own source files. Dense
#             antialiased small text on white: the class where palette decode
#             (`decode::read_pal_indices`) shows up at all.
#   ui      — a Quick Look render of a synthetic application window: flat
#             colour panels, 1px rules, table grids, buttons, a monospace code
#             block. The class that measured WORST against dav1d.
#
# All three at 16:9, one pixel count (default 1024x576), YUV420 8-bit, single
# tile, at four quality points (q20/q40/q70/q90 in avifenc's scale). The low-q
# end is not optional — that is where block counts and palette use are highest.
#
# HONEST LIMITS
#   * `text` and `ui` are RENDERS, not screenshots (this box has no screen
#     recording permission). They are real font rasterisation and real flat-panel
#     UI geometry, but they are not a sample of production screenshots.
#   * ONE source image per class. A per-class claim from n=1 source is a claim
#     about that image, and should be reported that way.
#   * avifenc's default tiling is 1 tile at every size here, so nothing in this
#     corpus exercises tile parallelism. That is a control, not a property of
#     production AVIFs (see mk_size_ladder.sh's note).
#
# PREREQUISITES
#   avifenc, avifdec (libavif CLI)      — encode + the photo source decode
#   qlmanage, sips                      — macOS built-ins, render + crop/scale
#   cargo build --release --example avif_to_ivf
#
# USAGE
#   scripts/perf/mk_content_classes.sh [outdir] [WxH] [frames]
# Default outdir $HOME/tmp/ctxtl, size 1024x576, 200 frames per IVF.
set -euo pipefail

ROOT="${1:-$HOME/tmp/ctxtl}"
WH="${2:-1024x576}"
FRAMES="${3:-200}"
W=${WH%x*}; H=${WH#*x}
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SRC="$ROOT/src"; VEC="$ROOT/vec"; IVF="$ROOT/ivf"; LOG="$ROOT/log"
mkdir -p "$SRC" "$VEC" "$IVF" "$LOG"

# --- 1. photo source --------------------------------------------------------
PHOTO_AVIF="$REPO/test-vectors/bench/photo_4k.avif"
[ -s "$PHOTO_AVIF" ] || { echo "missing $PHOTO_AVIF (gitignored — copy it in)" >&2; exit 3; }
[ -s "$SRC/photo4k.png" ] || nice -n 19 avifdec --png-compress 1 "$PHOTO_AVIF" "$SRC/photo4k.png" >"$LOG/avifdec.log" 2>&1

# --- 2. screen-text source: Quick Look render of this repo's own source ------
if [ ! -s "$SRC/code.txt.png" ]; then
  cat "$REPO/src/ctx.rs" "$REPO/src/env.rs" > "$SRC/code.txt"
  qlmanage -t -s 2048 -o "$SRC" "$SRC/code.txt" >"$LOG/ql_text.log" 2>&1
fi

# --- 3. screen-UI source: Quick Look render of a synthetic app window --------
if [ ! -s "$SRC/ui.html.png" ]; then
cat > "$SRC/ui.html" <<'HTMLEOF'
<html><head><meta charset="utf-8"><style>
body{margin:0;font-family:-apple-system,Helvetica,Arial;background:#f2f3f5;color:#1d1d1f}
.tb{height:44px;background:linear-gradient(#fbfbfd,#e8e8ed);border-bottom:1px solid #c6c6c8;display:flex;align-items:center;padding:0 12px}
.dot{width:12px;height:12px;border-radius:6px;margin-right:8px}
.side{position:absolute;left:0;top:44px;width:230px;bottom:0;background:#e9eaee;border-right:1px solid #c6c6c8}
.row{padding:6px 14px;font-size:13px;border-bottom:1px solid #dfe0e4}
.main{position:absolute;left:231px;top:44px;right:0;bottom:0;background:#fff;padding:14px}
table{border-collapse:collapse;font-size:12px;width:100%;margin-bottom:12px}
td,th{border:1px solid #d8d8dc;padding:4px 8px}
th{background:#f4f4f7;text-align:left}
.btn{display:inline-block;background:#0a68ff;color:#fff;border-radius:5px;padding:5px 14px;font-size:12px;margin-right:6px}
.btn2{display:inline-block;background:#e6e6eb;color:#1d1d1f;border-radius:5px;padding:5px 14px;font-size:12px;margin-right:6px}
pre{background:#f6f6f8;border:1px solid #e0e0e4;padding:8px;font-family:Menlo,monospace;font-size:11px;margin:0 0 12px 0}
.bar{height:10px;background:#dfe0e4;border-radius:5px;margin:3px 0}
.bar>i{display:block;height:10px;background:#34c759;border-radius:5px}
</style></head><body>
<div class="tb"><div class="dot" style="background:#ff5f57"></div><div class="dot" style="background:#febc2e"></div><div class="dot" style="background:#28c840"></div><b style="font-size:13px">Registration Inspector &mdash; per-site census</b></div>
<div class="side">
<div class="row">Overview</div><div class="row">Call sites</div><div class="row" style="background:#0a68ff;color:#fff">ctx.rs:99:27</div>
<div class="row">loopfilter.rs:566</div><div class="row">cdef_arm.rs:622</div><div class="row">mc.rs:121:61</div><div class="row">recon.rs:2380</div>
<div class="row">decode.rs:1997</div><div class="row">env.rs:141</div><div class="row">lf_mask.rs:314</div><div class="row">picture.rs:589</div>
<div class="row">Thread map</div><div class="row">Widening budget</div><div class="row">Conflict pairs</div><div class="row">Footprints</div>
<div class="row">Settings</div><div class="row">Export&hellip;</div>
</div>
<div class="main">
<div style="margin-bottom:10px"><span class="btn">Run census</span><span class="btn2">Compare</span><span class="btn2">Reset</span><span class="btn2">Export TSV</span></div>
<table><tr><th>site</th><th>regs/frame</th><th>share</th><th>mean extent</th><th>writer</th><th>gap 0 B</th></tr>
<tr><td>ctx.rs:99:27</td><td>2,534,988</td><td>43.9%</td><td>3.3 B</td><td>yes</td><td>36</td></tr>
<tr><td>loopfilter.rs:566</td><td>3,835,042</td><td>33.6%</td><td>90.7 B</td><td>no</td><td>0</td></tr>
<tr><td>cdef_arm.rs:622:9</td><td>1,863,648</td><td>16.3%</td><td>16 B</td><td>yes</td><td>0</td></tr>
<tr><td>cdef_apply.rs:104</td><td>669,376</td><td>5.9%</td><td>8 B</td><td>no</td><td>0</td></tr>
<tr><td>mc.rs:121:61</td><td>181,042,110</td><td>&mdash;</td><td>2.4 MB</td><td>no</td><td>0</td></tr>
<tr><td>picture.rs:589:26</td><td>4,096</td><td>0.0%</td><td>4,096 B</td><td>no</td><td>0</td></tr>
</table>
<pre>CaseSet::&lt;32,false&gt;::many(
    [(&amp;t.l, t_dim.lh, 1), (ta, t_dim.lw, 0)],
    [bh4 as usize, bw4 as usize],
    [by4 as usize, bx4 as usize],
    |case, (dir, lw_lh, dir_index)| {
        case.set_disjoint(&amp;dir.tx_intra, lw_lh as i8);
        case.set_disjoint(&amp;dir.mode, y_mode_nofilt);
        case.set_disjoint(&amp;dir.pal_sz, pal_sz[0]);
    },
);</pre>
<table><tr><th>gap bucket</th><th>0 B</th><th>&le;4 B</th><th>&le;64 B</th><th>&le;256 B</th><th>&le;4 KiB</th></tr>
<tr><td>foreign WRITE</td><td>36</td><td>263</td><td>1,204</td><td>9,881</td><td>60,552</td></tr>
<tr><td>foreign READ</td><td>0</td><td>12</td><td>340</td><td>4,102</td><td>221,904</td></tr></table>
<table><tr><th>thread</th><th>claims</th><th>overlaps</th><th>lost scans</th><th>utilisation</th></tr>
<tr><td>tc 0</td><td>1,425,881</td><td>0</td><td>4.0%</td><td><div class="bar"><i style="width:92%"></i></div></td></tr>
<tr><td>tc 1</td><td>1,401,204</td><td>0</td><td>4.1%</td><td><div class="bar"><i style="width:89%"></i></div></td></tr>
<tr><td>tc 2</td><td>1,388,410</td><td>0</td><td>3.8%</td><td><div class="bar"><i style="width:88%"></i></div></td></tr>
<tr><td>tc 3</td><td>1,377,002</td><td>0</td><td>4.4%</td><td><div class="bar"><i style="width:87%"></i></div></td></tr>
<tr><td>tc 4</td><td>1,352,119</td><td>0</td><td>5.1%</td><td><div class="bar"><i style="width:85%"></i></div></td></tr>
<tr><td>tc 5</td><td>1,340,776</td><td>0</td><td>5.4%</td><td><div class="bar"><i style="width:84%"></i></div></td></tr>
</table>
<pre>reconciliation: probe_bounds 11,401,399 == probe_sites 11,401,399   lost=0
mutable_overlaps 0   (407,046 immutable-vs-immutable overlaps seen)
teeth: +-4096 B widening moved mutable_overlaps 0 -&gt; 874</pre>
</div></body></html>
HTMLEOF
  qlmanage -t -s 2048 -o "$SRC" "$SRC/ui.html" >"$LOG/ql_ui.log" 2>&1
fi

# --- 4. crop to 16:9 and scale to the target ---------------------------------
# `qlmanage -t` renders a square page; take a 16:9 band from the TOP (that is
# where the rendered content is) rather than sips's default centre crop.
mk_src() { # <in.png> <out.png> <cropOffsetTop>
  local in=$1 out=$2 off=$3
  [ -s "$out" ] && return 0
  cp "$in" "$out.tmp.png"
  sips -c 1152 2048 --cropOffset "$off" 0 "$out.tmp.png" --out "$out.crop.png" >/dev/null
  sips -Z "$W" "$out.crop.png" --out "$out" >/dev/null
  rm -f "$out.tmp.png" "$out.crop.png"
}
mk_src "$SRC/ui.html.png"  "$SRC/ui_${WH}.png"   44
mk_src "$SRC/code.txt.png" "$SRC/text_${WH}.png" 0
if [ ! -s "$SRC/photo_${WH}.png" ]; then
  cp "$SRC/photo4k.png" "$SRC/photo_${WH}.png.tmp.png"
  sips -Z "$W" "$SRC/photo_${WH}.png.tmp.png" --out "$SRC/photo_${WH}.png" >/dev/null
  rm -f "$SRC/photo_${WH}.png.tmp.png"
fi

# --- 5. encode + IVF --------------------------------------------------------
B="${AVIF_TO_IVF:-$REPO/target/release/examples/avif_to_ivf}"
[ -x "$B" ] || { echo "build it first: cargo build --release --example avif_to_ivf" >&2; exit 3; }
for cls in photo text ui; do
  for q in 20 40 70 90; do
    v="C${cls}_${WH}_q${q}"
    if [ ! -s "$VEC/$v.avif" ]; then
      nice -n 19 avifenc -s 6 -q "$q" -y 420 -d 8 -j 8 --ignore-exif --ignore-xmp \
        "$SRC/${cls}_${WH}.png" "$VEC/$v.avif" >"$LOG/enc_$v.log" 2>&1
    fi
    [ -s "$IVF/$v.ivf" ] || nice -n 19 "$B" "$VEC/$v.avif" "$FRAMES" "$IVF/$v.ivf"
    printf '%s\t%s\n' "$v" "$(stat -f%z "$VEC/$v.avif")"
  done
done

echo
echo "NEXT, before ANY timing: prove every vector decodes bit-identically to"
echo "dav1d — scripts/perf/content_md5.sh (writes a by-name md5 table)."
