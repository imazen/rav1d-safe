// pan_zoom: natural-motion I420 clip from one large I420 still (see make_natural_streams.sh).
//   pan_zoom SRC.yuv SRC_W SRC_H OUT.yuv W H FRAMES
// Each frame is a W x H window sampled bilinearly (sub-pixel) from the source: a slow
// drifting pan with a gentle zoom (1.00 -> 1.06) plus deterministic sensor-like noise
// (sigma ~1.5 luma), so inter prediction has real fractional motion and non-zero residual.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static uint32_t rng = 12345;
static double gauss(void) { // sum of 4 uniforms, ~N(0, 0.58)
    double s = 0;
    for (int i = 0; i < 4; i++) { rng = rng * 1664525u + 1013904223u; s += (rng >> 8) / 16777216.0; }
    return (s - 2.0) * 1.73;
}
static double samp(const uint8_t *p, int pw, int ph, double x, double y) {
    if (x < 0) x = 0; if (y < 0) y = 0;
    if (x > pw - 1.001) x = pw - 1.001; if (y > ph - 1.001) y = ph - 1.001;
    int x0 = (int)x, y0 = (int)y; double fx = x - x0, fy = y - y0;
    const uint8_t *r0 = p + (size_t)y0 * pw + x0, *r1 = r0 + pw;
    return (r0[0] * (1 - fx) + r0[1] * fx) * (1 - fy) + (r1[0] * (1 - fx) + r1[1] * fx) * fy;
}
int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr, "usage: pan_zoom SRC SW SH OUT W H FRAMES\n"); return 2; }
    int sw = atoi(argv[2]), sh = atoi(argv[3]), W = atoi(argv[5]), H = atoi(argv[6]), N = atoi(argv[7]);
    size_t ys = (size_t)sw * sh, cs = ys / 4;
    uint8_t *src = malloc(ys * 3 / 2); FILE *f = fopen(argv[1], "rb");
    if (!f || fread(src, 1, ys * 3 / 2, f) != ys * 3 / 2) { fprintf(stderr, "read %s\n", argv[1]); return 1; }
    fclose(f);
    FILE *o = fopen(argv[4], "wb");
    uint8_t *Y = malloc((size_t)W * H), *C = malloc((size_t)W * H / 4);
    for (int n = 0; n < N; n++) {
        double t = N > 1 ? (double)n / (N - 1) : 0, z = 1.0 + 0.06 * t;     // zoom: window grows
        double vw = W * z, vh = H * z;
        double cx = sw * 0.5 + (sw * 0.5 - vw * 0.5 - 8) * (0.8 * sin(1.7 * t) ) ;
        double cy = sh * 0.5 + (sh * 0.5 - vh * 0.5 - 8) * (0.6 * sin(1.1 * t + 0.5));
        double ox = cx - vw / 2, oy = cy - vh / 2, sx = vw / W, sy = vh / H;
        for (int y = 0; y < H; y++) for (int x = 0; x < W; x++) {
            double v = samp(src, sw, sh, ox + (x + .5) * sx, oy + (y + .5) * sy) + 1.5 * gauss();
            Y[(size_t)y * W + x] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)(v + .5);
        }
        fwrite(Y, 1, (size_t)W * H, o);
        for (int pl = 0; pl < 2; pl++) {
            const uint8_t *cp = src + ys + pl * cs;
            for (int y = 0; y < H / 2; y++) for (int x = 0; x < W / 2; x++) {
                double v = samp(cp, sw / 2, sh / 2, (ox + (2 * x + 1) * sx) / 2, (oy + (2 * y + 1) * sy) / 2) + 0.7 * gauss();
                C[(size_t)y * (W / 2) + x] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)(v + .5);
            }
            fwrite(C, 1, (size_t)W * H / 4, o);
        }
    }
    fclose(o); return 0;
}
