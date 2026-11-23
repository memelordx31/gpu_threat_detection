/* ============================================================================
 * auth_score.cu
 * GPU-Accelerated Authentication Threat Detection project
 * combined per-event scoring kernel with a built-in CPU analogue for comparison.
 *
 * Combines the per-event outputs of the three detectors into one score:
 *
 *   det1 : auth_lateral    (float32 lateral-movement score)
 *   det2 : auth_fanout     (float32 fan-out score)
 *   det3 : auth_failburst  (float32 failed-logon burst score)
 * (any DSCR file works, uint32 or float32, e.g. the legacy hash detectors)
 *
 * The detectors produce values on unrelated scales, so each channel is first
 * rank-normalised to its empirical CDF:
 *
 *   r_c(i) = #{ k : x_c(k) < x_c(i) } / n         (ties share a rank, 0 = lowest)
 *
 * and then combined with weights held in __constant__ memory:
 *
 *   score(i) = (w1*r1 + w2*r2 + w3*r3) / (w1 + w2 + w3)          in [0, 1)
 *
 * GPU pipeline: channel c on stream c: H2D -> D2D copy -> thrust::sort ->
 * rank kernel (binary search). stream0 waits on the other two streams'
 * events, then the combine kernel writes the score and reduces, per block in
 * __shared__ memory, an alert count (score >= --alert), the max score and a
 * 16-bin score histogram. D2H on stream0.
 *
 * Input files use the DSCR header written by all three detectors:
 *   uint32 magic 'DSCR', uint32 dtype (0 = uint32, 1 = float32), uint32 n
 *
 * Build: nvcc -O2 -arch=sm_75 -Xcompiler -fopenmp auth_score.cu \
 *             -o auth_score -lgomp
 * Run:   ./auth_score --det1 lateral.bin --det2 fanout.bin --det3 burst.bin \
 *             --weights 1,1,1 --alert 0.95 --runs 6 --out combined.bin
 * ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <algorithm>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#define SCR_MAGIC  0x52435344u  
#define DTYPE_U32  0u
#define DTYPE_F32  1u
#define NCHAN      3
#define HIST_BINS  16

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ------------------------- command-line parameters ------------------------ */
static const char  *in_path[NCHAN] = { "det1.bin", "det2.bin", "det3.bin" };
static const char  *out_path       = NULL;
static unsigned int nthreads       = 256;
static unsigned int num_runs       = 6;
static int          cpu_only       = 0;
static int          skip_cpu       = 0;

/* ------------------- constant memory: combination parameters -------------- */
struct ScoreParams {
    float w[NCHAN];
    float inv_wsum;
    float alert;      /* alert threshold on the combined score */
};
__constant__ ScoreParams c_sp;
static ScoreParams h_sp = { { 1.0f, 1.0f, 1.0f }, 1.0f / 3.0f, 0.95f };

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const int has = (i + 1 < argc);
        if      (strcmp(argv[i], "--det1") == 0 && has)    in_path[0] = argv[++i];
        else if (strcmp(argv[i], "--det2") == 0 && has)    in_path[1] = argv[++i];
        else if (strcmp(argv[i], "--det3") == 0 && has)    in_path[2] = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && has)     out_path = argv[++i];
        else if (strcmp(argv[i], "--alert") == 0 && has)   h_sp.alert = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && has) nthreads = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && has)    num_runs = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu-only") == 0)       cpu_only = 1;
        else if (strcmp(argv[i], "--no-cpu") == 0)         skip_cpu = 1;
        else if (strcmp(argv[i], "--weights") == 0 && has) {
            if (sscanf(argv[++i], "%f,%f,%f", &h_sp.w[0], &h_sp.w[1], &h_sp.w[2]) != 3) {
                fprintf(stderr, "Error: --weights expects w1,w2,w3\n");
                exit(EXIT_FAILURE);
            }
        } else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--det1 P] [--det2 P] [--det3 P] [--out P] "
                            "[--weights w1,w2,w3] [--alert F] [--threads N] "
                            "[--runs N] [--cpu-only] [--no-cpu]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    const float wsum = h_sp.w[0] + h_sp.w[1] + h_sp.w[2];
    if (h_sp.w[0] < 0 || h_sp.w[1] < 0 || h_sp.w[2] < 0 || wsum <= 0.0f) {
        fprintf(stderr, "Error: weights must be >= 0 with a positive sum\n");
        exit(EXIT_FAILURE);
    }
    h_sp.inv_wsum = 1.0f / wsum;
    if (nthreads == 0 || nthreads > 1024 || (nthreads & (nthreads - 1)) != 0) {
        fprintf(stderr, "Error: --threads must be a power of two in [1, 1024]\n");
        exit(EXIT_FAILURE);
    }
    if (num_runs == 0) num_runs = 1;
    if (cpu_only) skip_cpu = 0;
}

/* ------------------------------ file I/O ---------------------------------- */
/* Map a float to a uint32 with the same ordering (handles negatives). */
static inline unsigned int float_order_key(float f)
{
    unsigned int u;
    memcpy(&u, &f, sizeof(u));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* Reads a DSCR file into ordered uint32 sort keys. Returns n. */
static unsigned int read_channel(const char *path, unsigned int **keys, int pinned)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    unsigned int hdr[3];
    if (fread(hdr, sizeof(hdr), 1, fp) != 1 || hdr[0] != SCR_MAGIC ||
        (hdr[1] != DTYPE_U32 && hdr[1] != DTYPE_F32) || hdr[2] == 0) {
        fprintf(stderr, "Error: '%s' is not a DSCR detector-score file\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned int n = hdr[2];
    const size_t bytes = (size_t)n * sizeof(unsigned int);
    if (pinned) CUDA_CHECK(cudaHostAlloc((void **)keys, bytes, cudaHostAllocDefault));
    else        *keys = (unsigned int *)malloc(bytes);
    if (!*keys || fread(*keys, sizeof(unsigned int), n, fp) != n) {
        fprintf(stderr, "Error: '%s' is truncated or allocation failed\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    if (hdr[1] == DTYPE_F32) {
        for (unsigned int i = 0; i < n; i++) {
            float f;
            memcpy(&f, &(*keys)[i], sizeof(f));
            (*keys)[i] = float_order_key(f);
        }
    }
    return n;
}

static void write_scores(const char *path, const float *s, unsigned int n)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Error: cannot write '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    const unsigned int hdr[3] = { SCR_MAGIC, DTYPE_F32, n };
    fwrite(hdr, sizeof(hdr), 1, fp);
    fwrite(s, sizeof(float), n, fp);
    fclose(fp);
}

/* ------------------------------ shared math ------------------------------- */
__host__ __device__ __forceinline__
unsigned int lower_bound_u32(const unsigned int *a, unsigned int n, unsigned int v)
{
    unsigned int lo = 0, hi = n;
    while (lo < hi) {
        const unsigned int mid = lo + ((hi - lo) >> 1);
        if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__host__ __device__ __forceinline__
float ecdf_rank(unsigned int below, unsigned int n)
{
    return (float)((double)below / (double)n);
}

/* explicit rounding on the device so GPU and CPU produce identical bits */
__host__ __device__ __forceinline__
float combine3(float r0, float r1, float r2, const ScoreParams &p)
{
#ifdef __CUDA_ARCH__
    float acc = __fmul_rn(p.w[0], r0);
    acc = __fadd_rn(acc, __fmul_rn(p.w[1], r1));
    acc = __fadd_rn(acc, __fmul_rn(p.w[2], r2));
    return __fmul_rn(acc, p.inv_wsum);
#else
    volatile float a = p.w[0] * r0;     
    a = a + p.w[1] * r1;
    a = a + p.w[2] * r2;
    return a * p.inv_wsum;
#endif
}

__host__ __device__ __forceinline__
unsigned int hist_bin(float s)
{
    int b = (int)(s * HIST_BINS);
    return (unsigned int)(b < 0 ? 0 : (b >= HIST_BINS ? HIST_BINS - 1 : b));
}

/* ------------------------------ GPU kernels ------------------------------- */
__global__ void rank_kernel(const unsigned int *keys, const unsigned int *sorted,
                            float *rank, const unsigned int n)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n)
        rank[tid] = ecdf_rank(lower_bound_u32(sorted, n, keys[tid]), n);
}

__global__ void combine_kernel(const float *r0, const float *r1, const float *r2,
                               float *score, unsigned int *alert_count,
                               unsigned int *max_bits, unsigned int *hist,
                               const unsigned int n)
{
    /* shared memory: block alert count + max (tree reduction), histogram */
    extern __shared__ unsigned int smem[];
    unsigned int *s_alert = smem;
    unsigned int *s_max   = smem + blockDim.x;
    __shared__ unsigned int s_hist[HIST_BINS];

    if (threadIdx.x < HIST_BINS)
        s_hist[threadIdx.x] = 0;
    __syncthreads();

    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    float s = 0.0f;
    unsigned int alert = 0;
    if (tid < n) {
        s = combine3(r0[tid], r1[tid], r2[tid], c_sp);
        score[tid] = s;
        alert = (s >= c_sp.alert) ? 1u : 0u;
        atomicAdd(&s_hist[hist_bin(s)], 1u);
    }
    s_alert[threadIdx.x] = alert;
    s_max[threadIdx.x]   = __float_as_uint(s);     /* s >= 0 */
    __syncthreads();

    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_alert[threadIdx.x] += s_alert[threadIdx.x + stride];
            s_max[threadIdx.x]    = max(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (s_alert[0]) atomicAdd(alert_count, s_alert[0]);
        atomicMax(max_bits, s_max[0]);
    }
    if (threadIdx.x < HIST_BINS && s_hist[threadIdx.x])
        atomicAdd(&hist[threadIdx.x], s_hist[threadIdx.x]);
}

/* ------------------------------- GPU driver ------------------------------- */
struct GpuBuffers {
    unsigned int *d_keys[NCHAN], *d_sorted[NCHAN];
    float        *d_rank[NCHAN], *d_score;
    unsigned int *d_alert, *d_max, *d_hist;
    unsigned int *h_small;                      /* pinned: alert, max, hist */
    cudaStream_t  s[NCHAN];
    cudaEvent_t   ev_start, ev_rank[NCHAN], ev_combined, ev_stop;
};

struct Summary {
    unsigned int alerts;
    float        max_score;
    unsigned int hist[HIST_BINS];
};

static void gpu_setup(GpuBuffers &b, unsigned int n)
{
    for (int c = 0; c < NCHAN; c++) {
        CUDA_CHECK(cudaMalloc(&b.d_keys[c],   (size_t)n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&b.d_sorted[c], (size_t)n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&b.d_rank[c],   (size_t)n * sizeof(float)));
        CUDA_CHECK(cudaStreamCreate(&b.s[c]));
        CUDA_CHECK(cudaEventCreate(&b.ev_rank[c]));
    }
    CUDA_CHECK(cudaMalloc(&b.d_score, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b.d_alert, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&b.d_max,   sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&b.d_hist,  HIST_BINS * sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc((void **)&b.h_small, (2 + HIST_BINS) * sizeof(unsigned int),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaEventCreate(&b.ev_start));
    CUDA_CHECK(cudaEventCreate(&b.ev_combined));
    CUDA_CHECK(cudaEventCreateWithFlags(&b.ev_stop, cudaEventBlockingSync));
}

static void gpu_teardown(GpuBuffers &b)
{
    for (int c = 0; c < NCHAN; c++) {
        cudaFree(b.d_keys[c]); cudaFree(b.d_sorted[c]); cudaFree(b.d_rank[c]);
        cudaStreamDestroy(b.s[c]); cudaEventDestroy(b.ev_rank[c]);
    }
    cudaFree(b.d_score); cudaFree(b.d_alert); cudaFree(b.d_max); cudaFree(b.d_hist);
    cudaFreeHost(b.h_small);
    cudaEventDestroy(b.ev_start); cudaEventDestroy(b.ev_combined);
    cudaEventDestroy(b.ev_stop);
}

static void gpu_run(unsigned int *const h_keys[NCHAN], float *h_out, unsigned int n,
                    GpuBuffers &b, Summary *sum, float *t_total, float *t_kernels)
{
    const size_t w = (size_t)n * sizeof(unsigned int);
    const unsigned int blocks = (n + nthreads - 1) / nthreads;

    CUDA_CHECK(cudaEventRecord(b.ev_start, b.s[0]));
    for (int c = 1; c < NCHAN; c++)
        CUDA_CHECK(cudaStreamWaitEvent(b.s[c], b.ev_start, 0));

    /* each channel is ranked independently on its own stream */
    for (int c = 0; c < NCHAN; c++) {
        thrust::device_ptr<unsigned int> sorted(b.d_sorted[c]);
        CUDA_CHECK(cudaMemcpyAsync(b.d_keys[c], h_keys[c], w, cudaMemcpyHostToDevice, b.s[c]));
        CUDA_CHECK(cudaMemcpyAsync(b.d_sorted[c], b.d_keys[c], w,
                                   cudaMemcpyDeviceToDevice, b.s[c]));
        thrust::sort(thrust::cuda::par.on(b.s[c]), sorted, sorted + n);
        rank_kernel<<<blocks, nthreads, 0, b.s[c]>>>(b.d_keys[c], b.d_sorted[c],
                                                     b.d_rank[c], n);
        CUDA_CHECK(cudaEventRecord(b.ev_rank[c], b.s[c]));
    }
    for (int c = 1; c < NCHAN; c++)
        CUDA_CHECK(cudaStreamWaitEvent(b.s[0], b.ev_rank[c], 0));

    CUDA_CHECK(cudaMemsetAsync(b.d_alert, 0, sizeof(unsigned int), b.s[0]));
    CUDA_CHECK(cudaMemsetAsync(b.d_max,   0, sizeof(unsigned int), b.s[0]));
    CUDA_CHECK(cudaMemsetAsync(b.d_hist,  0, HIST_BINS * sizeof(unsigned int), b.s[0]));
    combine_kernel<<<blocks, nthreads, 2 * nthreads * sizeof(unsigned int), b.s[0]>>>(
        b.d_rank[0], b.d_rank[1], b.d_rank[2], b.d_score,
        b.d_alert, b.d_max, b.d_hist, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(b.ev_combined, b.s[0]));

    CUDA_CHECK(cudaMemcpyAsync(h_out, b.d_score, (size_t)n * sizeof(float),
                               cudaMemcpyDeviceToHost, b.s[0]));
    CUDA_CHECK(cudaMemcpyAsync(&b.h_small[0], b.d_alert, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, b.s[0]));
    CUDA_CHECK(cudaMemcpyAsync(&b.h_small[1], b.d_max, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, b.s[0]));
    CUDA_CHECK(cudaMemcpyAsync(&b.h_small[2], b.d_hist, HIST_BINS * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, b.s[0]));
    CUDA_CHECK(cudaEventRecord(b.ev_stop, b.s[0]));
    CUDA_CHECK(cudaEventSynchronize(b.ev_stop));

    cudaEventElapsedTime(t_total,   b.ev_start, b.ev_stop);
    cudaEventElapsedTime(t_kernels, b.ev_start, b.ev_combined);  /* incl. H2D */
    sum->alerts = b.h_small[0];
    memcpy(&sum->max_score, &b.h_small[1], sizeof(float));
    memcpy(sum->hist, &b.h_small[2], sizeof(sum->hist));
}

/* ------------------------------- CPU analogue ----------------------------- */
static double cpu_run(unsigned int *const keys[NCHAN], float *out, unsigned int n,
                      std::vector<unsigned int> sorted[NCHAN],
                      std::vector<float> rank[NCHAN], Summary *sum)
{
    auto t0 = std::chrono::steady_clock::now();
    for (int c = 0; c < NCHAN; c++) {
        sorted[c].assign(keys[c], keys[c] + n);
        std::sort(sorted[c].begin(), sorted[c].end());
        rank[c].resize(n);
        const unsigned int *srt = sorted[c].data();
        const unsigned int *k   = keys[c];
        float *r = rank[c].data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (long long i = 0; i < (long long)n; i++)
            r[i] = ecdf_rank(lower_bound_u32(srt, n, k[i]), n);
    }

    unsigned int alerts = 0;
    float best = 0.0f;
    const float *r0 = rank[0].data(), *r1 = rank[1].data(), *r2 = rank[2].data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:alerts) reduction(max:best)
#endif
    for (long long i = 0; i < (long long)n; i++) {
        const float s = combine3(r0[i], r1[i], r2[i], h_sp);
        out[i] = s;
        if (s >= h_sp.alert) alerts++;
        if (s > best) best = s;
    }
    auto t1 = std::chrono::steady_clock::now();

    /* histogram is a reporting extra, kept outside the timed region */
    memset(sum->hist, 0, sizeof(sum->hist));
    for (unsigned int i = 0; i < n; i++) sum->hist[hist_bin(out[i])]++;
    sum->alerts = alerts;
    sum->max_score = best;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/* ---------------------------------- main ---------------------------------- */
int main(int argc, char **argv)
{
    parse_args(argc, argv);
    const int use_gpu = !cpu_only;

    printf("=== Authentication threat detection: combined per-event scoring ===\n");
    if (use_gpu) {
        struct cudaDeviceProp prop;
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        printf("GPU: %s | ", prop.name);
    }
    int cpu_threads = 1;
#ifdef _OPENMP
    cpu_threads = omp_get_max_threads();
#endif
    printf("CPU threads: %d | weights: %.2f, %.2f, %.2f | alert >= %.3f | runs: %u\n",
           cpu_threads, h_sp.w[0], h_sp.w[1], h_sp.w[2], h_sp.alert, num_runs);

    unsigned int *keys[NCHAN];
    unsigned int n = 0;
    for (int c = 0; c < NCHAN; c++) {
        const unsigned int nc = read_channel(in_path[c], &keys[c], use_gpu);
        if (c == 0) n = nc;
        else if (nc != n) {
            fprintf(stderr, "Error: '%s' has %u events but '%s' has %u; all detectors "
                            "must score the same event slice.\n",
                    in_path[c], nc, in_path[0], n);
            return EXIT_FAILURE;
        }
        printf("  channel %d: %s\n", c + 1, in_path[c]);
    }
    printf("Events: %u\n\n", n);

    float *gpu_out = NULL;
    float *cpu_out = !skip_cpu ? (float *)malloc((size_t)n * sizeof(float)) : NULL;
    Summary gs = {}, cs = {};
    double g_tot = 0, g_best = 0, g_kern = 0, c_tot = 0, c_best = 0;

    if (use_gpu) {
        CUDA_CHECK(cudaHostAlloc((void **)&gpu_out, (size_t)n * sizeof(float),
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaMemcpyToSymbol(c_sp, &h_sp, sizeof(ScoreParams)));
        GpuBuffers b;
        gpu_setup(b, n);
        float t, tk;
        gpu_run(keys, gpu_out, n, b, &gs, &t, &tk);                  /* warmup */
        for (unsigned int r = 0; r < num_runs; r++) {
            gpu_run(keys, gpu_out, n, b, &gs, &t, &tk);
            g_tot += t; g_kern += tk;
            if (r == 0 || t < g_best) g_best = t;
        }
        gpu_teardown(b);
        g_tot /= num_runs; g_kern /= num_runs;
        printf("GPU (avg of %u): total %.3f ms | best %.3f ms | to-combine %.3f ms\n",
               num_runs, g_tot, g_best, g_kern);
    }

    if (!skip_cpu) {
        std::vector<unsigned int> sorted[NCHAN];
        std::vector<float> rank[NCHAN];
        for (unsigned int r = 0; r < num_runs; r++) {
            const double t = cpu_run(keys, cpu_out, n, sorted, rank, &cs);
            c_tot += t;
            if (r == 0 || t < c_best) c_best = t;
        }
        c_tot /= num_runs;
        printf("CPU (avg of %u): total %.3f ms | best %.3f ms\n", num_runs, c_tot, c_best);
    }

    double max_diff = 0.0;
    int match = -1;
    if (use_gpu && !skip_cpu) {
        for (unsigned int i = 0; i < n; i++) {
            const double d = fabs((double)gpu_out[i] - (double)cpu_out[i]);
            if (d > max_diff) max_diff = d;
        }
        match = (max_diff <= 1e-6 && gs.alerts == cs.alerts) ? 1 : 0;
        printf("Verification: GPU vs CPU max |diff| = %.3e, alerts %u vs %u -> %s\n",
               max_diff, gs.alerts, cs.alerts, match ? "MATCH" : "MISMATCH");
        printf("Speedup (avg end-to-end incl. transfers): %.2fx\n", c_tot / g_tot);
    }

    const Summary &S = use_gpu ? gs : cs;
    const float *final_out = use_gpu ? gpu_out : cpu_out;
    printf("\nAlerts (score >= %.3f): %u (%.3f%%) | max score %.4f\n",
           h_sp.alert, S.alerts, 100.0 * S.alerts / n, S.max_score);
    printf("Score histogram (shared-memory reduction):\n");
    for (int b = 0; b < HIST_BINS; b++)
        printf("  [%.4f, %.4f) %10u\n", (double)b / HIST_BINS, (double)(b + 1) / HIST_BINS,
               S.hist[b]);

    if (out_path) {
        write_scores(out_path, final_out, n);
        printf("Wrote combined per-event scores to %s\n", out_path);
    }

    printf("RESULT {\"detector\":\"score\",\"n\":%u,\"gpu_ms\":%.4f,\"gpu_best_ms\":%.4f,"
           "\"cpu_ms\":%.4f,\"cpu_best_ms\":%.4f,\"cpu_threads\":%d,\"alerts\":%u,"
           "\"max_score\":%.6f,\"max_abs_diff\":%.6g,\"match\":%d}\n",
           n, use_gpu ? g_tot : -1.0, use_gpu ? g_best : -1.0,
           skip_cpu ? -1.0 : c_tot, skip_cpu ? -1.0 : c_best, cpu_threads,
           S.alerts, S.max_score, max_diff, match);

    for (int c = 0; c < NCHAN; c++) {
        if (use_gpu) cudaFreeHost(keys[c]); else free(keys[c]);
    }
    if (gpu_out) cudaFreeHost(gpu_out);
    free(cpu_out);
    if (use_gpu) cudaDeviceReset();
    return (match == 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
