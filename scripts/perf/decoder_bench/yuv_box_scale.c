// yuv_box_scale SRC.yuv SW SH K OUT.yuv : integer box-filter downscale of an I420 stream by K (SW, SH divisible by 2K).
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
static void scale(const uint8_t *s, int sw, int sh, int k, uint8_t *d) {
    int dw = sw / k, dh = sh / k;
    for (int y = 0; y < dh; y++) for (int x = 0; x < dw; x++) {
        int a = 0;
        for (int j = 0; j < k; j++) for (int i = 0; i < k; i++) a += s[(size_t)(y * k + j) * sw + x * k + i];
        d[(size_t)y * dw + x] = (a + k * k / 2) / (k * k);
    }
}
int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: yuv_box_scale SRC SW SH K OUT\n"); return 2; }
    int sw = atoi(argv[2]), sh = atoi(argv[3]), k = atoi(argv[4]);
    size_t fs = (size_t)sw * sh * 3 / 2, dfs = (size_t)(sw / k) * (sh / k) * 3 / 2;
    uint8_t *in = malloc(fs), *out = malloc(dfs);
    FILE *f = fopen(argv[1], "rb"), *o = fopen(argv[5], "wb"); long n = 0;
    while (fread(in, 1, fs, f) == fs) {
        scale(in, sw, sh, k, out);
        scale(in + (size_t)sw * sh, sw / 2, sh / 2, k, out + (size_t)(sw / k) * (sh / k));
        scale(in + (size_t)sw * sh * 5 / 4, sw / 2, sh / 2, k, out + (size_t)(sw / k) * (sh / k) * 5 / 4);
        fwrite(out, 1, dfs, o); n++;
    }
    fprintf(stderr, "%ld frames\n", n); return 0;
}
