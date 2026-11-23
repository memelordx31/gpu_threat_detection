/* ============================================================================
 * auth_failburst.cu
 * GPU-Accelerated Authentication Threat Detection project
 * failed-logon burst detector component with a built-in CPU analogue for comparison.
 *
 * For every authentication event the detector counts how many failed logons
 * the same entity (source user by default) produced inside two trailing time
 * windows ending at that event:
 *
 *   short window  (default   60 s) : rapid password spraying / guessing
 *   long  window  (default 3600 s) : slow, persistent guessing
 *
 * and turns the counts into a per-event burst score:
 *
 *   score = w_short*log2(1+fails_short) + w_long*log2(1+fails_long)
 *         + w_ratio*(fails_long / attempts_long)          (0 if fails_long < min)
 *
 * The score is assigned whether or not the current event itself failed, so a
 * successful logon right after a burst of failures (brute-force
 * "it finally worked" pattern) scores highly.
 *
 * GPU pipeline (one pass):
 *   H2D (pinned, async) -> build 64-bit (entity<<32 | time) keys
 *   -> thrust::sort_by_key -> gather fail flags -> thrust::inclusive_scan
 *   -> short window kernel on stream0 || long window kernel on stream1
 *   -> score kernel (params in __constant__, per-block __shared__ reduction
 *      of alert count and max score) -> scatter back to file order -> D2H
 *
 * CPU analogue: the same algorithm,
 * OpenMP-parallel over events when compiled with -fopenmp.
 *
 * Input:  events.bin written by run_pipeline.py (preprocessed LANL cyber1
 *         auth.txt slice, same record order as the slice the other two
 *         detectors read, so per-event outputs line up by index).
 * Output: --out PATH  ->  DSCR float32 per-event burst scores (file order)
 *
 * Dataset: https://csr.lanl.gov/data/cyber1/
 * Needs auth_common.cuh in the same directory.
 * Build: nvcc -O2 -arch=sm_75 -Xcompiler -fopenmp auth_failburst.cu \
 *             -o auth_failburst -lgomp
 * Run:   ./auth_failburst --events events.bin --key user --short 60 \
 *             --long 3600 --threads 256 --runs 6 --out det3.bin
 *        ./auth_failburst --events events.bin --cpu-only   (no GPU needed)
 * ==========================================================================*/

#include <math.h>

#include <algorithm>
#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

#include "auth_common.cuh"

/* ------------------------- command-line parameters ------------------------ */
static const char  *events_path = "events.bin";
static const char  *out_path    = NULL;
static const char  *key_name    = "user";   /* user | src | dst               */
static unsigned int nthreads    = 256;
static unsigned int num_runs    = 6;
static int          cpu_only    = 0;
static int          skip_cpu    = 0;

/* -------------------- constant memory: detector parameters ----------------
 * Unlike the static __constant__ masks in auth_scan_memtypes.cu these are set
 * at run time from the command line with cudaMemcpyToSymbol. */
struct BurstParams {
    unsigned int win_short;     /* seconds                                  */
    unsigned int win_long;      /* seconds                                  */
    unsigned int burst_alert;   /* fails in short window that raise alert   */
    unsigned int min_fails;     /* fails in long window needed for score>0  */
    float        w_short;
    float        w_long;
    float        w_ratio;
};
__constant__ BurstParams c_params;
static BurstParams h_params = { 60u, 3600u, 5u, 1u, 1.0f, 0.5f, 1.0f };

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [--events PATH] [--out PATH] [--key user|src|dst]\n"
        "          [--short SEC] [--long SEC] [--alert N] [--min-fails N]\n"
        "          [--w-short F] [--w-long F] [--w-ratio F]\n"
        "          [--threads N] [--runs N] [--cpu-only] [--no-cpu]\n", prog);
}

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const int has = (i + 1 < argc);
        if      (strcmp(argv[i], "--events") == 0 && has)    events_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && has)       out_path = argv[++i];
        else if (strcmp(argv[i], "--key") == 0 && has)       key_name = argv[++i];
        else if (strcmp(argv[i], "--short") == 0 && has)     h_params.win_short = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--long") == 0 && has)      h_params.win_long = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--alert") == 0 && has)     h_params.burst_alert = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--min-fails") == 0 && has) h_params.min_fails = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--w-short") == 0 && has)   h_params.w_short = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--w-long") == 0 && has)    h_params.w_long = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--w-ratio") == 0 && has)   h_params.w_ratio = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && has)   nthreads = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && has)      num_runs = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu-only") == 0)         cpu_only = 1;
        else if (strcmp(argv[i], "--no-cpu") == 0)           skip_cpu = 1;
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            usage(argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    /* the shared-memory tree reduction needs a power-of-two block size */
    if (nthreads == 0 || nthreads > 1024 || (nthreads & (nthreads - 1)) != 0) {
        fprintf(stderr, "Error: --threads must be a power of two in [1, 1024]\n");
        exit(EXIT_FAILURE);
    }
    if (h_params.win_short == 0 || h_params.win_long == 0) {
        fprintf(stderr, "Error: window lengths must be >= 1 second\n");
        exit(EXIT_FAILURE);
    }
    if (num_runs == 0) num_runs = 1;
    if (cpu_only) skip_cpu = 0;
}

/* ------------------------------ shared math ------------------------------- */
__host__ __device__ __forceinline__
float burst_formula(unsigned int fs, unsigned int fl, unsigned int tl,
                    const BurstParams &p)
{
    if (fl < p.min_fails || fl == 0u)
        return 0.0f;
    return p.w_short * log2f(1.0f + (float)fs)
         + p.w_long  * log2f(1.0f + (float)fl)
         + p.w_ratio * ((float)fl / (float)tl);
}

/* first index with a[idx] >= v (lower) / > v (upper) in a sorted array */
__host__ __device__ __forceinline__
unsigned int lower_bound_u64(const unsigned long long *a, unsigned int n,
                             unsigned long long v)
{
    unsigned int lo = 0, hi = n;
    while (lo < hi) {
        const unsigned int mid = lo + ((hi - lo) >> 1);
        if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

__host__ __device__ __forceinline__
unsigned int upper_bound_u64(const unsigned long long *a, unsigned int n,
                             unsigned long long v)
{
    unsigned int lo = 0, hi = n;
    while (lo < hi) {
        const unsigned int mid = lo + ((hi - lo) >> 1);
        if (a[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* Window (t - win, t] for the entity encoded in the high 32 bits of k.
 * F is the exclusive prefix sum of fail flags in sorted order (size n+1). */
__host__ __device__ __forceinline__
void window_counts(const unsigned long long *k64, const unsigned int *F,
                   unsigned int n, unsigned int j, unsigned int win,
                   unsigned int *fails, unsigned int *total)
{
    const unsigned long long k    = k64[j];
    const unsigned int       t    = (unsigned int)(k & 0xFFFFFFFFull);
    const unsigned long long base = k & 0xFFFFFFFF00000000ull;
    const unsigned int       t0   = (t + 1u > win) ? (t + 1u - win) : 0u;
    const unsigned int lo = lower_bound_u64(k64, n, base | t0);
    const unsigned int hi = upper_bound_u64(k64, n, k);   /* includes same-second ties */
    *fails = F[hi] - F[lo];
    *total = hi - lo;
}

/* ------------------------------ GPU kernels ------------------------------- */
__global__ void build_keys(const unsigned int *key, const unsigned int *time,
                           unsigned long long *k64, unsigned int *perm,
                           const unsigned int n)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n) {
        k64[tid]  = ((unsigned long long)key[tid] << 32) | time[tid];
        perm[tid] = tid;
    }
}

__global__ void gather_fail(const unsigned int *perm, const unsigned char *fail,
                            unsigned int *fail_sorted, const unsigned int n)
{
    const unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (tid < n)
        fail_sorted[tid] = fail[perm[tid]];
}

__global__ void burst_window(const unsigned long long *k64, const unsigned int *F,
                             const unsigned int n, const unsigned int win,
                             unsigned int *fails_out, unsigned int *total_out)
{
    const unsigned int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (j < n)
        window_counts(k64, F, n, j, win, &fails_out[j], &total_out[j]);
}

__global__ void burst_score(const unsigned int *perm,
                            const unsigned int *fs,  /* fails, short window   */
                            const unsigned int *fl,  /* fails, long window    */
                            const unsigned int *tl,  /* attempts, long window */
                            float *score_out,        /* file order            */
                            unsigned int *alert_count,
                            unsigned int *max_score_bits,
                            const unsigned int n)
{
    extern __shared__ unsigned int smem[];
    unsigned int *s_alert = smem;
    unsigned int *s_max   = smem + blockDim.x;

    const unsigned int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int alert = 0;
    float s = 0.0f;
    if (j < n) {
        s = burst_formula(fs[j], fl[j], tl[j], c_params);
        score_out[perm[j]] = s;                       /* scatter to file order */
        alert = (fs[j] >= c_params.burst_alert) ? 1u : 0u;
    }
    s_alert[threadIdx.x] = alert;
    s_max[threadIdx.x]   = __float_as_uint(s);       /* s >= 0: bit order == value order */
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
        atomicMax(max_score_bits, s_max[0]);
    }
}

/* ------------------------------- GPU driver ------------------------------- */
struct GpuBuffers {
    unsigned int       *d_key, *d_time, *d_perm, *d_fsorted, *d_F;
    unsigned char      *d_fail;
    unsigned long long *d_k64;
    unsigned int       *d_fs, *d_ts, *d_fl, *d_tl, *d_alert, *d_max;
    float              *d_score;
    unsigned int       *h_small;   /* pinned: [0]=alerts, [1]=max bits */
    cudaStream_t        s0, s1;
    cudaEvent_t         ev_start, ev_h2d, ev_sorted, ev_long, ev_scored, ev_stop;
};

struct GpuTiming { float total, h2d, sort_scan, windows_score, d2h; };

static void gpu_setup(GpuBuffers &b, unsigned int n)
{
    const size_t w = (size_t)n * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&b.d_key, w));
    CUDA_CHECK(cudaMalloc(&b.d_time, w));
    CUDA_CHECK(cudaMalloc(&b.d_perm, w));
    CUDA_CHECK(cudaMalloc(&b.d_fsorted, w));
    CUDA_CHECK(cudaMalloc(&b.d_F, w + sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&b.d_fail, n));
    CUDA_CHECK(cudaMalloc(&b.d_k64, (size_t)n * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&b.d_fs, w));
    CUDA_CHECK(cudaMalloc(&b.d_ts, w));
    CUDA_CHECK(cudaMalloc(&b.d_fl, w));
    CUDA_CHECK(cudaMalloc(&b.d_tl, w));
    CUDA_CHECK(cudaMalloc(&b.d_alert, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&b.d_max, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&b.d_score, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaHostAlloc((void **)&b.h_small, 2 * sizeof(unsigned int),
                             cudaHostAllocDefault));
    CUDA_CHECK(cudaStreamCreate(&b.s0));
    CUDA_CHECK(cudaStreamCreate(&b.s1));
    CUDA_CHECK(cudaEventCreate(&b.ev_start));
    CUDA_CHECK(cudaEventCreate(&b.ev_h2d));
    CUDA_CHECK(cudaEventCreate(&b.ev_sorted));
    CUDA_CHECK(cudaEventCreate(&b.ev_long));
    CUDA_CHECK(cudaEventCreate(&b.ev_scored));
    CUDA_CHECK(cudaEventCreateWithFlags(&b.ev_stop, cudaEventBlockingSync));
}

static void gpu_teardown(GpuBuffers &b)
{
    cudaFree(b.d_key); cudaFree(b.d_time); cudaFree(b.d_perm);
    cudaFree(b.d_fsorted); cudaFree(b.d_F); cudaFree(b.d_fail);
    cudaFree(b.d_k64); cudaFree(b.d_fs); cudaFree(b.d_ts);
    cudaFree(b.d_fl); cudaFree(b.d_tl); cudaFree(b.d_alert);
    cudaFree(b.d_max); cudaFree(b.d_score); cudaFreeHost(b.h_small);
    cudaStreamDestroy(b.s0); cudaStreamDestroy(b.s1);
    cudaEventDestroy(b.ev_start); cudaEventDestroy(b.ev_h2d);
    cudaEventDestroy(b.ev_sorted); cudaEventDestroy(b.ev_long);
    cudaEventDestroy(b.ev_scored); cudaEventDestroy(b.ev_stop);
}

/* One full detection pass, H2D through D2H, timed with events. */
static GpuTiming gpu_run(const Events &ev, const unsigned int *h_key,
                         float *h_out, GpuBuffers &b,
                         unsigned int *alerts, float *max_score)
{
    const unsigned int n = ev.n;
    const size_t w = (size_t)n * sizeof(unsigned int);
    const unsigned int blocks = (n + nthreads - 1) / nthreads;
    thrust::device_ptr<unsigned long long> k64(b.d_k64);
    thrust::device_ptr<unsigned int>       perm(b.d_perm);
    thrust::device_ptr<unsigned int>       fsorted(b.d_fsorted);
    thrust::device_ptr<unsigned int>       F(b.d_F);

    CUDA_CHECK(cudaEventRecord(b.ev_start, b.s0));
    CUDA_CHECK(cudaMemcpyAsync(b.d_key,  h_key,   w, cudaMemcpyHostToDevice, b.s0));
    CUDA_CHECK(cudaMemcpyAsync(b.d_time, ev.time, w, cudaMemcpyHostToDevice, b.s0));
    CUDA_CHECK(cudaMemcpyAsync(b.d_fail, ev.fail, n, cudaMemcpyHostToDevice, b.s0));
    CUDA_CHECK(cudaEventRecord(b.ev_h2d, b.s0));

    /* group each entity's events together, in time order */
    build_keys<<<blocks, nthreads, 0, b.s0>>>(b.d_key, b.d_time, b.d_k64, b.d_perm, n);
    thrust::sort_by_key(thrust::cuda::par.on(b.s0), k64, k64 + n, perm);
    gather_fail<<<blocks, nthreads, 0, b.s0>>>(b.d_perm, b.d_fail, b.d_fsorted, n);
    CUDA_CHECK(cudaMemsetAsync(b.d_F, 0, sizeof(unsigned int), b.s0));
    thrust::inclusive_scan(thrust::cuda::par.on(b.s0), fsorted, fsorted + n, F + 1);
    CUDA_CHECK(cudaEventRecord(b.ev_sorted, b.s0));

    /* short and long windows concurrently on two streams */
    CUDA_CHECK(cudaStreamWaitEvent(b.s1, b.ev_sorted, 0));
    burst_window<<<blocks, nthreads, 0, b.s0>>>(b.d_k64, b.d_F, n, h_params.win_short,
                                                b.d_fs, b.d_ts);
    burst_window<<<blocks, nthreads, 0, b.s1>>>(b.d_k64, b.d_F, n, h_params.win_long,
                                                b.d_fl, b.d_tl);
    CUDA_CHECK(cudaEventRecord(b.ev_long, b.s1));
    CUDA_CHECK(cudaStreamWaitEvent(b.s0, b.ev_long, 0));

    CUDA_CHECK(cudaMemsetAsync(b.d_alert, 0, sizeof(unsigned int), b.s0));
    CUDA_CHECK(cudaMemsetAsync(b.d_max,   0, sizeof(unsigned int), b.s0));
    burst_score<<<blocks, nthreads, 2 * nthreads * sizeof(unsigned int), b.s0>>>(
        b.d_perm, b.d_fs, b.d_fl, b.d_tl, b.d_score, b.d_alert, b.d_max, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(b.ev_scored, b.s0));

    CUDA_CHECK(cudaMemcpyAsync(h_out, b.d_score, (size_t)n * sizeof(float),
                               cudaMemcpyDeviceToHost, b.s0));
    CUDA_CHECK(cudaMemcpyAsync(&b.h_small[0], b.d_alert, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, b.s0));
    CUDA_CHECK(cudaMemcpyAsync(&b.h_small[1], b.d_max, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, b.s0));
    CUDA_CHECK(cudaEventRecord(b.ev_stop, b.s0));
    CUDA_CHECK(cudaEventSynchronize(b.ev_stop));

    GpuTiming t;
    cudaEventElapsedTime(&t.total,         b.ev_start,  b.ev_stop);
    cudaEventElapsedTime(&t.h2d,           b.ev_start,  b.ev_h2d);
    cudaEventElapsedTime(&t.sort_scan,     b.ev_h2d,    b.ev_sorted);
    cudaEventElapsedTime(&t.windows_score, b.ev_sorted, b.ev_scored);
    cudaEventElapsedTime(&t.d2h,           b.ev_scored, b.ev_stop);

    *alerts = b.h_small[0];
    unsigned int bits = b.h_small[1];
    memcpy(max_score, &bits, sizeof(float));
    return t;
}

/* ------------------------------- CPU analogue ----------------------------- */
struct CpuScratch {
    std::vector<std::pair<unsigned long long, unsigned int> > kp;
    std::vector<unsigned long long> k64;
    std::vector<unsigned int>       F;
};

static double cpu_run(const Events &ev, const unsigned int *key, float *out,
                      CpuScratch &s, unsigned int *alerts, float *max_score)
{
    const unsigned int n = ev.n;
    auto t0 = std::chrono::steady_clock::now();

    s.kp.resize(n);
    s.k64.resize(n);
    s.F.resize((size_t)n + 1);
    for (unsigned int i = 0; i < n; i++)
        s.kp[i] = std::make_pair(((unsigned long long)key[i] << 32) | ev.time[i], i);
    std::sort(s.kp.begin(), s.kp.end());

    s.F[0] = 0;
    for (unsigned int j = 0; j < n; j++) {
        s.k64[j]   = s.kp[j].first;
        s.F[j + 1] = s.F[j] + ev.fail[s.kp[j].second];
    }

    unsigned int n_alert = 0;
    float best = 0.0f;
    const unsigned long long *k = s.k64.data();
    const unsigned int *F = s.F.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:n_alert) reduction(max:best)
#endif
    for (long long jj = 0; jj < (long long)n; jj++) {
        const unsigned int j = (unsigned int)jj;
        unsigned int fs, ts, fl, tl;
        window_counts(k, F, n, j, h_params.win_short, &fs, &ts);
        window_counts(k, F, n, j, h_params.win_long,  &fl, &tl);
        const float sc = burst_formula(fs, fl, tl, h_params);
        out[s.kp[j].second] = sc;
        if (fs >= h_params.burst_alert) n_alert++;
        if (sc > best) best = sc;
    }

    auto t1 = std::chrono::steady_clock::now();
    *alerts = n_alert;
    *max_score = best;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/* ---------------------------------- main ---------------------------------- */
int main(int argc, char **argv)
{
    parse_args(argc, argv);
    const int use_gpu = !cpu_only;

    printf("=== Authentication threat detection: failed-logon burst detector ===\n");
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
    printf("CPU threads: %d | key: %s | windows: %us / %us | threads: %u | runs: %u\n",
           cpu_threads, key_name, h_params.win_short, h_params.win_long,
           nthreads, num_runs);

    Events ev = read_events(events_path, use_gpu);
    const unsigned int n = ev.n;
    const unsigned int *key =
        strcmp(key_name, "user") == 0 ? ev.user :
        strcmp(key_name, "src")  == 0 ? ev.src  :
        strcmp(key_name, "dst")  == 0 ? ev.dst  : NULL;
    if (!key) {
        fprintf(stderr, "Error: --key must be user, src or dst\n");
        return EXIT_FAILURE;
    }
    unsigned int n_fail = 0;
    for (unsigned int i = 0; i < n; i++) n_fail += ev.fail[i];
    printf("Loaded %u events from %s (%u failed logons, %.2f%%)\n\n",
           n, events_path, n_fail, 100.0 * n_fail / n);

    float *gpu_out = use_gpu ? (float *)host_alloc((size_t)n * sizeof(float), 1) : NULL;
    float *cpu_out = !skip_cpu ? (float *)malloc((size_t)n * sizeof(float)) : NULL;

    /* ----------------------------- GPU passes ------------------------------ */
    double g_tot = 0, g_best = 0, g_h2d = 0, g_sort = 0, g_win = 0, g_d2h = 0;
    unsigned int g_alerts = 0;
    float g_max = 0.0f;
    if (use_gpu) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_params, &h_params, sizeof(BurstParams)));
        GpuBuffers b;
        gpu_setup(b, n);
        gpu_run(ev, key, gpu_out, b, &g_alerts, &g_max);          /* warmup */
        for (unsigned int r = 0; r < num_runs; r++) {
            GpuTiming t = gpu_run(ev, key, gpu_out, b, &g_alerts, &g_max);
            g_tot += t.total; g_h2d += t.h2d; g_sort += t.sort_scan;
            g_win += t.windows_score; g_d2h += t.d2h;
            if (r == 0 || t.total < g_best) g_best = t.total;
        }
        gpu_teardown(b);
        g_tot /= num_runs; g_h2d /= num_runs; g_sort /= num_runs;
        g_win /= num_runs; g_d2h /= num_runs;
        printf("GPU (avg of %u): total %.3f ms | best %.3f ms\n", num_runs, g_tot, g_best);
        printf("    H2D %.3f | sort+scan %.3f | windows+score %.3f | D2H %.3f ms\n",
               g_h2d, g_sort, g_win, g_d2h);
    }

    /* ----------------------------- CPU passes ------------------------------ */
    double c_tot = 0, c_best = 0;
    unsigned int c_alerts = 0;
    float c_max = 0.0f;
    if (!skip_cpu) {
        CpuScratch s;
        for (unsigned int r = 0; r < num_runs; r++) {
            const double t = cpu_run(ev, key, cpu_out, s, &c_alerts, &c_max);
            c_tot += t;
            if (r == 0 || t < c_best) c_best = t;
        }
        c_tot /= num_runs;
        printf("CPU (avg of %u): total %.3f ms | best %.3f ms\n", num_runs, c_tot, c_best);
    }

    /* --------------------------- verification ------------------------------ */
    double max_diff = 0.0;
    int match = -1;   /* -1: not compared */
    if (use_gpu && !skip_cpu) {
        for (unsigned int i = 0; i < n; i++) {
            const double d = fabs((double)gpu_out[i] - (double)cpu_out[i]);
            if (d > max_diff) max_diff = d;
        }
        match = (max_diff <= 1e-4 && g_alerts == c_alerts) ? 1 : 0;
        printf("Verification: GPU vs CPU max |diff| = %.3e, alerts %u vs %u -> %s\n",
               max_diff, g_alerts, c_alerts, match ? "MATCH" : "MISMATCH");
        printf("Speedup (avg end-to-end incl. transfers): %.2fx\n", c_tot / g_tot);
    }

    const float *final_out = use_gpu ? gpu_out : cpu_out;
    const unsigned int alerts = use_gpu ? g_alerts : c_alerts;
    const float max_score     = use_gpu ? g_max : c_max;
    unsigned int nonzero = 0;
    for (unsigned int i = 0; i < n; i++) nonzero += (final_out[i] > 0.0f);
    printf("\nEvents with burst score > 0: %u (%.2f%%)\n", nonzero, 100.0 * nonzero / n);
    printf("Short-window alerts (>= %u fails / %us): %u | max score %.3f\n",
           h_params.burst_alert, h_params.win_short, alerts, max_score);

    if (out_path) {
        write_scores_f32(out_path, final_out, n);
        printf("Wrote per-event burst scores to %s\n", out_path);
    }

    printf("RESULT {\"detector\":\"failburst\",\"n\":%u,\"gpu_ms\":%.4f,"
           "\"gpu_best_ms\":%.4f,\"gpu_h2d_ms\":%.4f,\"gpu_sort_ms\":%.4f,"
           "\"gpu_window_score_ms\":%.4f,\"gpu_d2h_ms\":%.4f,\"cpu_ms\":%.4f,"
           "\"cpu_best_ms\":%.4f,\"cpu_threads\":%d,\"alerts\":%u,\"nonzero\":%u,"
           "\"max_abs_diff\":%.6g,\"match\":%d}\n",
           n, use_gpu ? g_tot : -1.0, use_gpu ? g_best : -1.0, g_h2d, g_sort, g_win,
           g_d2h, skip_cpu ? -1.0 : c_tot, skip_cpu ? -1.0 : c_best, cpu_threads,
           alerts, nonzero, max_diff, match);

    if (gpu_out) host_free(gpu_out, 1);
    free(cpu_out);
    free_events(ev, use_gpu);
    if (use_gpu) cudaDeviceReset();
    return (match == 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
