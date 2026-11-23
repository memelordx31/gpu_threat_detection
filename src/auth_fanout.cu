/* ============================================================================
 * auth_fanout.cu
 * GPU-Accelerated Authentication Threat Detection project
 * fan-out detector component with a built-in CPU analogue for comparison.
 *
 * Fan-out is one entity (a source computer by default, or a user) touching
 * many distinct destination computers in a short time. This may occur during enumeration or
 * credential sparying across the network. For every
 * authentication attempt with src != dst (success or failure), time is cut
 * into fixed buckets at two scales and the detector counts
 *
 *   D_S, D_L       : distinct destinations of the entity in its current short
 *                    (600 s) and long (3600 s) bucket
 *   mean_S, mean_L : the entity's average D_S / D_L over the buckets it was
 *                    active in (its own baseline)
 *
 * and every event (of any kind) gets its entity's current fan-out score:
 *
 *   score = w_short*log2(1+D_S) + w_long*log2(1+D_L)
 *         + w_rel*( max(0, log2((1+D_S)/(1+mean_S)))
 *                 + max(0, log2((1+D_L)/(1+mean_L))) )
 *
 * The relative terms dominate by default. Busy servers and heavy users that
 * always talk to many hosts are not flagged just for being busy; a host that
 * suddenly touches far more machines than it normally does is.
 *
 * GPU pipeline: H2D on stream0; short scale on stream0 || long scale on
 * stream1, each: build (entity-bucket << 32 | dst) keys ->
 * thrust::sort -> unique -> reduce_by_key (distinct dsts per entity-bucket)
 * -> reduce_by_key again (per-entity baseline);
 * then the score kernel (binary-search lookups, weights in __constant__,
 * per-block __shared__ reduction of alert count and max score) -> D2H.
 *
 * Build: nvcc -O2 -arch=sm_75 -Xcompiler -fopenmp auth_fanout.cu \
 *             -o auth_fanout -lgomp
 * Run:   ./auth_fanout --events events.bin --key src --short 600 --long 3600 \
 *             --runs 6 --out fanout.bin
 *        ./auth_fanout --events events.bin --cpu-only
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
#include <thrust/unique.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>

#include "auth_common.cuh"

#define NO_KEY 0xFFFFFFFFFFFFFFFFull   /* sentinel for events that don't count */

/* ------------------------- command-line parameters ------------------------ */
static const char  *events_path = "events.bin";
static const char  *out_path    = NULL;
static const char  *key_name    = "src";     /* src | user */
static unsigned int nthreads    = 256;
static unsigned int num_runs    = 6;
static int          cpu_only    = 0;
static int          skip_cpu    = 0;
static unsigned int n_counted   = 0;         /* events with src != dst */

/* ------------------- constant memory: detector parameters ----------------- */
struct FanParams {
    unsigned int b_short, b_long;      /* bucket widths, seconds            */
    unsigned int base_s, base_l;       /* first bucket index of the slice   */
    unsigned int nb_s, nb_l;           /* buckets per entity at each scale  */
    unsigned int fan_alert;            /* D_short that raises an alert      */
    float        w_short, w_long, w_rel;
};
__constant__ FanParams c_fp;
static FanParams h_fp = { 600u, 3600u, 0u, 0u, 1u, 1u, 10u, 0.25f, 0.25f, 1.0f };

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const int has = (i + 1 < argc);
        if      (strcmp(argv[i], "--events") == 0 && has)  events_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && has)     out_path = argv[++i];
        else if (strcmp(argv[i], "--key") == 0 && has)     key_name = argv[++i];
        else if (strcmp(argv[i], "--short") == 0 && has)   h_fp.b_short = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--long") == 0 && has)    h_fp.b_long = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--alert") == 0 && has)   h_fp.fan_alert = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--w-short") == 0 && has) h_fp.w_short = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--w-long") == 0 && has)  h_fp.w_long = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--w-rel") == 0 && has)   h_fp.w_rel = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && has) nthreads = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && has)    num_runs = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu-only") == 0)       cpu_only = 1;
        else if (strcmp(argv[i], "--no-cpu") == 0)         skip_cpu = 1;
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--events P] [--out P] [--key src|user] "
                            "[--short SEC] [--long SEC] [--alert N] [--w-short F] "
                            "[--w-long F] [--w-rel F] [--threads N] [--runs N] "
                            "[--cpu-only] [--no-cpu]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (!check_block_size(nthreads)) exit(EXIT_FAILURE);
    if (h_fp.b_short == 0 || h_fp.b_long == 0) {
        fprintf(stderr, "Error: bucket widths must be >= 1 second\n");
        exit(EXIT_FAILURE);
    }
    if (strcmp(key_name, "src") != 0 && strcmp(key_name, "user") != 0) {
        fprintf(stderr, "Error: --key must be src or user\n");
        exit(EXIT_FAILURE);
    }
    if (num_runs == 0) num_runs = 1;
    if (cpu_only) skip_cpu = 0;
}

/* ------------------------------ shared math ------------------------------- */
__host__ __device__ __forceinline__
unsigned int entity_bucket(unsigned int entity, unsigned int t, unsigned int width,
                           unsigned int base, unsigned int nb)
{
    return entity * nb + (t / width - base);
}

__host__ __device__ __forceinline__
float rel_term(unsigned int d, unsigned int sum, unsigned int nbk)
{
    const float mean = nbk ? (float)sum / (float)nbk : 0.0f;
    const float r = log2f((1.0f + (float)d) / (1.0f + mean));
    return r > 0.0f ? r : 0.0f;
}

__host__ __device__ __forceinline__
float fan_formula(unsigned int ds, unsigned int dl,
                  unsigned int sum_s, unsigned int nbk_s,
                  unsigned int sum_l, unsigned int nbk_l, const FanParams &p)
{
    if (ds == 0u && dl == 0u) return 0.0f;
    return p.w_short * log2f(1.0f + (float)ds) + p.w_long * log2f(1.0f + (float)dl)
         + p.w_rel * (rel_term(ds, sum_s, nbk_s) + rel_term(dl, sum_l, nbk_l));
}

__host__ __device__ __forceinline__
unsigned int lookup_count(const unsigned int *ids, const unsigned int *cnt,
                          unsigned int n, unsigned int v)
{
    const unsigned int i = find_dev(ids, n, v);
    return i < n ? cnt[i] : 0u;
}

struct hi32 {
    __host__ __device__ unsigned int operator()(unsigned long long x) const {
        return (unsigned int)(x >> 32);
    }
};
struct div_by {
    unsigned int d;
    __host__ __device__ unsigned int operator()(unsigned int x) const { return x / d; }
};

/* ------------------------------ GPU kernels ------------------------------- */
__global__ void fan_keys(const unsigned int *entity, const unsigned int *time,
                         const unsigned int *src, const unsigned int *dst,
                         const unsigned int width, const unsigned int base,
                         const unsigned int nb, unsigned long long *keys,
                         const unsigned int n)
{
    const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (i < n) {
        keys[i] = (src[i] != dst[i])
                ? ((unsigned long long)entity_bucket(entity[i], time[i], width, base, nb) << 32)
                  | dst[i]
                : NO_KEY;
    }
}

struct ScaleView {                 /* read-only lookup tables of one scale */
    const unsigned int *ids, *cnt;   /* distinct dsts per entity-bucket      */
    unsigned int ne;
    const unsigned int *ent_ids, *ent_sum, *ent_nbk;   /* per-entity baseline */
    unsigned int nent;
};

__host__ __device__ __forceinline__
void scale_lookup(const ScaleView &v, unsigned int eb, unsigned int e,
                  unsigned int *d, unsigned int *sum, unsigned int *nbk)
{
    *d = lookup_count(v.ids, v.cnt, v.ne, eb);
    const unsigned int ei = find_dev(v.ent_ids, v.nent, e);
    *sum = ei < v.nent ? v.ent_sum[ei] : 0u;
    *nbk = ei < v.nent ? v.ent_nbk[ei] : 0u;
}

__global__ void fan_score(const unsigned int *entity, const unsigned int *time,
                          const ScaleView vs, const ScaleView vl,
                          float *score, unsigned int *alert_count,
                          unsigned int *max_bits, const unsigned int n)
{
    extern __shared__ unsigned int smem[];
    unsigned int *s_alert = smem;
    unsigned int *s_max   = smem + blockDim.x;

    const unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int alert = 0;
    float s = 0.0f;
    if (i < n) {
        const unsigned int e = entity[i], t = time[i];
        unsigned int ds, ss, bs, dl, sl, bl;
        scale_lookup(vs, entity_bucket(e, t, c_fp.b_short, c_fp.base_s, c_fp.nb_s), e,
                     &ds, &ss, &bs);
        scale_lookup(vl, entity_bucket(e, t, c_fp.b_long, c_fp.base_l, c_fp.nb_l), e,
                     &dl, &sl, &bl);
        s = fan_formula(ds, dl, ss, bs, sl, bl, c_fp);
        score[i] = s;
        alert = (ds >= c_fp.fan_alert) ? 1u : 0u;
    }
    s_alert[threadIdx.x] = alert;
    s_max[threadIdx.x]   = __float_as_uint(s);
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
}

/* ------------------------------- GPU driver ------------------------------- */
struct Scale {                     /* device buffers for one bucket width */
    unsigned long long *d_keys;
    unsigned int *d_hi, *d_ids, *d_cnt, *d_ent, *d_ent_ids, *d_ent_sum, *d_ent_ids2,
                 *d_ent_nbk;
    unsigned int ne, nent;
    ScaleView view() const {
        ScaleView v = { d_ids, d_cnt, ne, d_ent_ids, d_ent_sum, d_ent_nbk, nent };
        return v;
    }
};

struct Gpu {
    unsigned int *d_entity, *d_time, *d_src, *d_dst;
    Scale sc[2];                       /* 0 = short on stream0, 1 = long on stream1 */
    unsigned int *d_alert, *d_max, *h_small;
    float *d_score;
    cudaStream_t s[2];
    cudaEvent_t ev_start, ev_h2d, ev_long, ev_stop;
};

static void gpu_setup(Gpu &g, unsigned int n)
{
    const size_t w = (size_t)n * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&g.d_entity, w)); CUDA_CHECK(cudaMalloc(&g.d_time, w));
    CUDA_CHECK(cudaMalloc(&g.d_src, w));    CUDA_CHECK(cudaMalloc(&g.d_dst, w));
    for (int c = 0; c < 2; c++) {
        Scale &sc = g.sc[c];
        CUDA_CHECK(cudaMalloc(&sc.d_keys, (size_t)n * 8));
        unsigned int **bufs[] = { &sc.d_hi, &sc.d_ids, &sc.d_cnt, &sc.d_ent, &sc.d_ent_ids,
                                  &sc.d_ent_sum, &sc.d_ent_ids2, &sc.d_ent_nbk };
        for (size_t b = 0; b < sizeof(bufs) / sizeof(bufs[0]); b++)
            CUDA_CHECK(cudaMalloc(bufs[b], w));
        CUDA_CHECK(cudaStreamCreate(&g.s[c]));
    }
    CUDA_CHECK(cudaMalloc(&g.d_alert, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&g.d_max, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&g.d_score, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaHostAlloc((void **)&g.h_small, 2 * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaEventCreate(&g.ev_start)); CUDA_CHECK(cudaEventCreate(&g.ev_h2d));
    CUDA_CHECK(cudaEventCreate(&g.ev_long));
    CUDA_CHECK(cudaEventCreateWithFlags(&g.ev_stop, cudaEventBlockingSync));
}

static void gpu_teardown(Gpu &g)
{
    cudaFree(g.d_entity); cudaFree(g.d_time); cudaFree(g.d_src); cudaFree(g.d_dst);
    for (int c = 0; c < 2; c++) {
        Scale &sc = g.sc[c];
        cudaFree(sc.d_keys); cudaFree(sc.d_hi); cudaFree(sc.d_ids); cudaFree(sc.d_cnt);
        cudaFree(sc.d_ent); cudaFree(sc.d_ent_ids); cudaFree(sc.d_ent_sum);
        cudaFree(sc.d_ent_ids2); cudaFree(sc.d_ent_nbk);
        cudaStreamDestroy(g.s[c]);
    }
    cudaFree(g.d_alert); cudaFree(g.d_max); cudaFree(g.d_score); cudaFreeHost(g.h_small);
    cudaEventDestroy(g.ev_start); cudaEventDestroy(g.ev_h2d);
    cudaEventDestroy(g.ev_long); cudaEventDestroy(g.ev_stop);
}

/* distinct destinations per entity-bucket at one scale, plus each entity's
 * baseline (sum and number of active buckets), all on one stream */
static void gpu_scale(Gpu &g, Scale &sc, cudaStream_t st, unsigned int width,
                      unsigned int base, unsigned int nb, unsigned int n)
{
    using thrust::device_ptr;
    const unsigned int blocks = (n + nthreads - 1) / nthreads;
    fan_keys<<<blocks, nthreads, 0, st>>>(g.d_entity, g.d_time, g.d_src, g.d_dst,
                                          width, base, nb, sc.d_keys, n);
    device_ptr<unsigned long long> k(sc.d_keys);
    thrust::sort(thrust::cuda::par.on(st), k, k + n);
    /* NO_KEY sorts last, so the first n_counted keys are the real ones */
    const unsigned int nu = (unsigned int)(thrust::unique(thrust::cuda::par.on(st), k,
                                                          k + n_counted) - k);
    device_ptr<unsigned int> hi(sc.d_hi), ids(sc.d_ids), cnt(sc.d_cnt);
    thrust::transform(thrust::cuda::par.on(st), k, k + nu, hi, hi32());
    sc.ne = (unsigned int)(thrust::reduce_by_key(thrust::cuda::par.on(st), hi, hi + nu,
        thrust::constant_iterator<unsigned int>(1), ids, cnt).first - ids);

    div_by dv; dv.d = nb;
    device_ptr<unsigned int> ent(sc.d_ent), ent_ids(sc.d_ent_ids), ent_sum(sc.d_ent_sum),
                             ent_ids2(sc.d_ent_ids2), ent_nbk(sc.d_ent_nbk);
    thrust::transform(thrust::cuda::par.on(st), ids, ids + sc.ne, ent, dv);
    sc.nent = (unsigned int)(thrust::reduce_by_key(thrust::cuda::par.on(st), ent,
        ent + sc.ne, cnt, ent_ids, ent_sum).first - ent_ids);
    thrust::reduce_by_key(thrust::cuda::par.on(st), ent, ent + sc.ne,
                          thrust::constant_iterator<unsigned int>(1), ent_ids2, ent_nbk);
}

struct Stats { unsigned int alerts; float max_score; unsigned int ne_s, ne_l; };

static float gpu_run(const Events &ev, const unsigned int *entity, float *h_out,
                     Gpu &g, Stats *st)
{
    const unsigned int n = ev.n;
    const size_t w = (size_t)n * sizeof(unsigned int);
    const unsigned int blocks = (n + nthreads - 1) / nthreads;
    cudaStream_t s0 = g.s[0], s1 = g.s[1];

    CUDA_CHECK(cudaEventRecord(g.ev_start, s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_entity, entity, w, cudaMemcpyHostToDevice, s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_time, ev.time, w, cudaMemcpyHostToDevice, s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_src, ev.src, w, cudaMemcpyHostToDevice, s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_dst, ev.dst, w, cudaMemcpyHostToDevice, s0));
    CUDA_CHECK(cudaEventRecord(g.ev_h2d, s0));

    /* long scale on stream1, short scale on stream0 */
    CUDA_CHECK(cudaStreamWaitEvent(s1, g.ev_h2d, 0));
    gpu_scale(g, g.sc[1], s1, h_fp.b_long, h_fp.base_l, h_fp.nb_l, n);
    CUDA_CHECK(cudaEventRecord(g.ev_long, s1));
    gpu_scale(g, g.sc[0], s0, h_fp.b_short, h_fp.base_s, h_fp.nb_s, n);

    CUDA_CHECK(cudaStreamWaitEvent(s0, g.ev_long, 0));
    CUDA_CHECK(cudaMemsetAsync(g.d_alert, 0, sizeof(unsigned int), s0));
    CUDA_CHECK(cudaMemsetAsync(g.d_max, 0, sizeof(unsigned int), s0));
    fan_score<<<blocks, nthreads, 2 * nthreads * sizeof(unsigned int), s0>>>(
        g.d_entity, g.d_time, g.sc[0].view(), g.sc[1].view(),
        g.d_score, g.d_alert, g.d_max, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_out, g.d_score, (size_t)n * sizeof(float),
                               cudaMemcpyDeviceToHost, s0));
    CUDA_CHECK(cudaMemcpyAsync(&g.h_small[0], g.d_alert, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, s0));
    CUDA_CHECK(cudaMemcpyAsync(&g.h_small[1], g.d_max, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, s0));
    CUDA_CHECK(cudaEventRecord(g.ev_stop, s0));
    CUDA_CHECK(cudaEventSynchronize(g.ev_stop));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, g.ev_start, g.ev_stop);
    st->alerts = g.h_small[0];
    memcpy(&st->max_score, &g.h_small[1], sizeof(float));
    st->ne_s = g.sc[0].ne; st->ne_l = g.sc[1].ne;
    return ms;
}

/* ------------------------------- CPU analogue ----------------------------- */
struct CpuScale {
    std::vector<unsigned int> ids, cnt, ent_ids, ent_sum, ent_nbk;
    ScaleView view() const {
        ScaleView v = { ids.data(), cnt.data(), (unsigned int)ids.size(), ent_ids.data(),
                        ent_sum.data(), ent_nbk.data(), (unsigned int)ent_ids.size() };
        return v;
    }
};

static void cpu_scale(const Events &ev, const unsigned int *entity, unsigned int width,
                      unsigned int base, unsigned int nb, std::vector<unsigned long long> &k,
                      CpuScale &sc)
{
    k.clear();
    for (unsigned int i = 0; i < ev.n; i++)
        if (ev.src[i] != ev.dst[i])
            k.push_back(pack64(entity_bucket(entity[i], ev.time[i], width, base, nb), ev.dst[i]));
    std::sort(k.begin(), k.end());
    k.erase(std::unique(k.begin(), k.end()), k.end());
    sc.ids.clear(); sc.cnt.clear(); sc.ent_ids.clear(); sc.ent_sum.clear(); sc.ent_nbk.clear();
    for (size_t q = 0; q < k.size(); q++) {
        const unsigned int eb = (unsigned int)(k[q] >> 32);
        if (sc.ids.empty() || sc.ids.back() != eb) { sc.ids.push_back(eb); sc.cnt.push_back(0); }
        sc.cnt.back()++;
    }
    for (size_t q = 0; q < sc.ids.size(); q++) {
        const unsigned int e = sc.ids[q] / nb;
        if (sc.ent_ids.empty() || sc.ent_ids.back() != e) {
            sc.ent_ids.push_back(e); sc.ent_sum.push_back(0); sc.ent_nbk.push_back(0);
        }
        sc.ent_sum.back() += sc.cnt[q];
        sc.ent_nbk.back()++;
    }
}

static double cpu_run(const Events &ev, const unsigned int *entity, float *out, Stats *st)
{
    auto t0 = std::chrono::steady_clock::now();
    std::vector<unsigned long long> k;
    CpuScale S, L;
    cpu_scale(ev, entity, h_fp.b_short, h_fp.base_s, h_fp.nb_s, k, S);
    cpu_scale(ev, entity, h_fp.b_long,  h_fp.base_l, h_fp.nb_l, k, L);
    const ScaleView vs = S.view(), vl = L.view();

    unsigned int alerts = 0;
    float best = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+:alerts) reduction(max:best)
#endif
    for (long long ii = 0; ii < (long long)ev.n; ii++) {
        const unsigned int i = (unsigned int)ii, e = entity[i], t = ev.time[i];
        unsigned int ds, ss, bs, dl, sl, bl;
        scale_lookup(vs, entity_bucket(e, t, h_fp.b_short, h_fp.base_s, h_fp.nb_s), e,
                     &ds, &ss, &bs);
        scale_lookup(vl, entity_bucket(e, t, h_fp.b_long, h_fp.base_l, h_fp.nb_l), e,
                     &dl, &sl, &bl);
        const float s = fan_formula(ds, dl, ss, bs, sl, bl, h_fp);
        out[i] = s;
        if (ds >= h_fp.fan_alert) alerts++;
        if (s > best) best = s;
    }
    auto t1 = std::chrono::steady_clock::now();
    st->alerts = alerts; st->max_score = best;
    st->ne_s = vs.ne; st->ne_l = vl.ne;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/* ---------------------------------- main ---------------------------------- */
int main(int argc, char **argv)
{
    parse_args(argc, argv);
    const int use_gpu = !cpu_only;

    printf("=== Authentication threat detection: fan-out detector ===\n");
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

    Events ev = read_events(events_path, use_gpu);
    const unsigned int n = ev.n;
    const unsigned int *entity = strcmp(key_name, "user") == 0 ? ev.user : ev.src;
    unsigned int max_entity = 0;
    for (unsigned int i = 0; i < n; i++) {
        max_entity = std::max(max_entity, entity[i]);
        n_counted += (ev.src[i] != ev.dst[i]);
    }

    h_fp.base_s = ev.tmin / h_fp.b_short;
    h_fp.base_l = ev.tmin / h_fp.b_long;
    h_fp.nb_s   = ev.tmax / h_fp.b_short - h_fp.base_s + 1;
    h_fp.nb_l   = ev.tmax / h_fp.b_long  - h_fp.base_l + 1;
    const unsigned long long eb_max =
        (unsigned long long)(max_entity + 1ull) * std::max(h_fp.nb_s, h_fp.nb_l);
    if (eb_max >= 0xFFFFFFFFull) {
        fprintf(stderr, "Error: %u entities x %u buckets does not fit 32 bits; "
                        "use a wider --short bucket or a shorter slice.\n",
                max_entity + 1, std::max(h_fp.nb_s, h_fp.nb_l));
        return EXIT_FAILURE;
    }
    printf("CPU threads: %d | key: %s | buckets: %us / %us | alert: D_short >= %u | runs: %u\n",
           cpu_threads, key_name, h_fp.b_short, h_fp.b_long, h_fp.fan_alert, num_runs);
    printf("Loaded %u events from %s, t = %u..%u\n\n", n, events_path, ev.tmin, ev.tmax);

    float *gpu_out = use_gpu ? (float *)host_alloc((size_t)n * sizeof(float), 1) : NULL;
    float *cpu_out = !skip_cpu ? (float *)malloc((size_t)n * sizeof(float)) : NULL;
    Stats gs = {}, cs = {};
    double g_tot = 0, g_best = 0, c_tot = 0, c_best = 0;

    if (use_gpu) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_fp, &h_fp, sizeof(FanParams)));
        Gpu g;
        gpu_setup(g, n);
        gpu_run(ev, entity, gpu_out, g, &gs);                          /* warmup */
        for (unsigned int r = 0; r < num_runs; r++) {
            const double t = gpu_run(ev, entity, gpu_out, g, &gs);
            g_tot += t;
            if (r == 0 || t < g_best) g_best = t;
        }
        gpu_teardown(g);
        g_tot /= num_runs;
        printf("GPU (avg of %u): total %.3f ms | best %.3f ms\n", num_runs, g_tot, g_best);
    }
    if (!skip_cpu) {
        for (unsigned int r = 0; r < num_runs; r++) {
            const double t = cpu_run(ev, entity, cpu_out, &cs);
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
        match = (max_diff <= 1e-4 && gs.alerts == cs.alerts);
        printf("Verification: GPU vs CPU max |diff| = %.3e, alerts %u vs %u -> %s\n",
               max_diff, gs.alerts, cs.alerts, match ? "MATCH" : "MISMATCH");
        printf("Speedup (avg end-to-end incl. transfers): %.2fx\n", c_tot / g_tot);
    }

    const Stats &S = use_gpu ? gs : cs;
    const float *final_out = use_gpu ? gpu_out : cpu_out;
    printf("\nActive entity-buckets: %u short, %u long | alerts (D_short >= %u): %u | "
           "max score %.3f\n", S.ne_s, S.ne_l, h_fp.fan_alert, S.alerts, S.max_score);
    if (out_path) {
        write_scores_f32(out_path, final_out, n);
        printf("Wrote per-event fan-out scores to %s\n", out_path);
    }

    printf("RESULT {\"detector\":\"fanout\",\"n\":%u,\"gpu_ms\":%.4f,\"gpu_best_ms\":%.4f,"
           "\"cpu_ms\":%.4f,\"cpu_best_ms\":%.4f,\"cpu_threads\":%d,\"alerts\":%u,"
           "\"max_abs_diff\":%.6g,\"match\":%d}\n",
           n, use_gpu ? g_tot : -1.0, use_gpu ? g_best : -1.0,
           skip_cpu ? -1.0 : c_tot, skip_cpu ? -1.0 : c_best, cpu_threads,
           S.alerts, max_diff, match);

    host_free(gpu_out, 1);
    free(cpu_out);
    free_events(ev, use_gpu);
    if (use_gpu) cudaDeviceReset();
    return (match == 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
