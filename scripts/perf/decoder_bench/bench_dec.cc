// Same-protocol in-process decode timing for dav1d and libgav1.
// Mirrors examples/profile_ivf.rs: IVF parsed into memory once; each pass creates a
// FRESH decoder (so creation/thread-spawn is counted for everyone), feeds every
// temporal unit, drains and releases every picture, destroys the decoder. Time is
// taken inside the process, around the pass only (no process startup, no file IO).
//   bench_dec <dav1d|gav1> <file.ivf> <threads> <passes> <avx2only 0|1> [mode tile|auto]
// mode tile: tile/post-filter threading only (dav1d max_frame_delay=1, libgav1 not frame-parallel).
// mode auto: each decoder's default parallelism (dav1d max_frame_delay=0 = auto, libgav1
//            frame_parallel with its own enqueue-until-TryAgain protocol).
// Prints one line per measured pass: "PASS <ms_per_frame> <frames>".
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
extern "C" {
#include "dav1d/dav1d.h"
}
#include "gav1/decoder.h"
extern "C" void dav1d_set_cpu_flags_mask(unsigned mask);  // declared in dav1d src/cpu.h (internal header)

struct Tu { const uint8_t* p; size_t n; };

static int pass_dav1d(const std::vector<Tu>& tus, int threads, bool autodelay) {
  Dav1dSettings s; dav1d_default_settings(&s);
  s.n_threads = threads; s.max_frame_delay = autodelay ? 0 : 1; s.logger.callback = nullptr;
  Dav1dContext* c = nullptr;
  if (dav1d_open(&c, &s) < 0) { fprintf(stderr, "dav1d_open failed\n"); exit(1); }
  int n = 0;
  auto drain_one = [&]() -> int {
    Dav1dPicture pic; memset(&pic, 0, sizeof pic);
    int r = dav1d_get_picture(c, &pic);
    if (r == 0) { n++; dav1d_picture_unref(&pic); return 0; }
    if (r != DAV1D_ERR(EAGAIN)) { fprintf(stderr, "get_picture error %d\n", r); exit(1); }
    return r;
  };
  for (const Tu& tu : tus) {
    Dav1dData d; memset(&d, 0, sizeof d);
    if (dav1d_data_wrap(&d, tu.p, tu.n, [](const uint8_t*, void*) {}, nullptr) < 0) exit(1);
    do {
      int r = dav1d_send_data(c, &d);
      if (r < 0 && r != DAV1D_ERR(EAGAIN)) { fprintf(stderr, "send_data error %d\n", r); exit(1); }
      drain_one();
    } while (d.sz > 0);
  }
  while (drain_one() == 0) {}
  dav1d_close(&c);
  return n;
}

static int pass_gav1_parallel(const std::vector<Tu>& tus, int threads) {
  libgav1::DecoderSettings s; s.threads = threads; s.frame_parallel = true; s.blocking_dequeue = true;
  s.release_input_buffer = [](void*, void*) {};  // inputs outlive the pass
  libgav1::Decoder dec;
  if (dec.Init(&s) != libgav1::kStatusOk) { fprintf(stderr, "gav1 init failed\n"); exit(1); }
  int n = 0; size_t i = 0;
  for (;;) {
    if (i < tus.size()) {
      auto st = dec.EnqueueFrame(tus[i].p, tus[i].n, 0, nullptr);
      if (st == libgav1::kStatusOk) { i++; continue; }
      if (st != libgav1::kStatusTryAgain) { fprintf(stderr, "enqueue error %s\n", libgav1::GetErrorString(st)); exit(1); }
    }
    const libgav1::DecoderBuffer* b = nullptr;
    auto st = dec.DequeueFrame(&b);
    if (st == libgav1::kStatusNothingToDequeue) { if (i >= tus.size()) break; continue; }
    if (st != libgav1::kStatusOk) { fprintf(stderr, "dequeue error %s\n", libgav1::GetErrorString(st)); exit(1); }
    if (b) n++;
  }
  return n;
}

static int pass_gav1(const std::vector<Tu>& tus, int threads) {
  libgav1::DecoderSettings s; s.threads = threads; s.frame_parallel = false;
  libgav1::Decoder dec;
  if (dec.Init(&s) != libgav1::kStatusOk) { fprintf(stderr, "gav1 init failed\n"); exit(1); }
  int n = 0;
  auto drain = [&]() {
    for (;;) {
      const libgav1::DecoderBuffer* b = nullptr;
      auto st = dec.DequeueFrame(&b);
      if (st == libgav1::kStatusNothingToDequeue) return;
      if (st != libgav1::kStatusOk) { fprintf(stderr, "dequeue error %s\n", libgav1::GetErrorString(st)); exit(1); }
      if (b) n++;
    }
  };
  for (const Tu& tu : tus) {
    for (;;) {
      auto st = dec.EnqueueFrame(tu.p, tu.n, 0, nullptr);
      if (st == libgav1::kStatusOk) break;
      if (st != libgav1::kStatusTryAgain) { fprintf(stderr, "enqueue error %s\n", libgav1::GetErrorString(st)); exit(1); }
      drain();
    }
    drain();
  }
  drain();
  return n;
}

int main(int argc, char** argv) {
  if (argc < 6) { fprintf(stderr, "usage: bench_dec <dav1d|gav1> <file> <threads> <passes> <avx2only>\n"); return 2; }
  std::string which = argv[1]; int threads = atoi(argv[3]), passes = atoi(argv[4]); bool avx2 = atoi(argv[5]);
  bool autom = argc > 6 && std::string(argv[6]) == "auto";
  FILE* f = fopen(argv[2], "rb"); if (!f) { perror("open"); return 1; }
  std::vector<uint8_t> data; { uint8_t b[65536]; size_t r; while ((r = fread(b, 1, sizeof b, f)) > 0) data.insert(data.end(), b, b + r); }
  fclose(f);
  size_t hl = data[6] | (data[7] << 8), pos = hl; std::vector<Tu> tus;
  while (pos + 12 <= data.size()) { uint32_t sz; memcpy(&sz, &data[pos], 4); tus.push_back({&data[pos + 12], sz}); pos += 12 + sz; }
  if (which == "dav1d" && avx2) dav1d_set_cpu_flags_mask(0xF);  // SSE2|SSSE3|SSE4.1|AVX2, no AVX-512
  auto run = [&]() { return which == "dav1d" ? pass_dav1d(tus, threads, autom) : (autom ? pass_gav1_parallel(tus, threads) : pass_gav1(tus, threads)); };
  run();  // warm-up pass, same as profile_ivf
  for (int i = 0; i < passes; i++) {
    auto t0 = std::chrono::steady_clock::now();
    int n = run();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("PASS %.6f %d\n", ms / (n ? n : 1), n);
  }
}
