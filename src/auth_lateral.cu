/* ============================================================================
 * auth_lateral.cu
 * GPU-Accelerated Authentication Threat Detection project
 * lateral-movement detector component with a built-in CPU analogue for comparison.
 *
 * Lateral movement shows up in authentication logs as a user reaching hosts
 * they have never used, and doing so in hops: an attacker lands on host B
 * through a new user to host edge and soon afterwards authenticates from B to
 * another host C they have also never touched. Using successful logons only:
 *
 *   new_edge(e) : e is the first time user(e) authenticated to dst(e) in the
 *                 slice, after the learning period, and src(e) != dst(e)
 *   hop(e)      : new_edge(e), and user(e) itself ARRIVED at src(e) through a
 *                 new edge within the last --hop-window seconds
 *   pop(d)      : distinct users that used host d during the learning period
 *                 (rarely used hosts are the more surprising targets)
 *
 *   score(e) = new_edge(e) * ( w_new + w_rare / (1 + pop(dst)) + w_hop * hop(e) )
 *
 *
 * GPU pipeline:
 *   H2D -> compact successful events (thrust::copy_if)
 *   stream1: (user,dst) edge dictionary   : sort + unique
 *   stream2: host popularity (learn period): sort + unique + reduce_by_key
 *   stream0: (edge id << 32 | time) keys -> sort_by_key -> new-edge flags ->
 *            arrivals (copy_if) -> score kernel: hop lookup by binary search,
 *            weights in __constant__, per-block __shared__ reduction of the
 *            new-edge / hop counts and the max score -> scatter -> D2H
 *
 * Build: nvcc -O2 -arch=sm_75 -Xcompiler -fopenmp auth_lateral.cu \
 *             -o auth_lateral -lgomp
 * Run:   ./auth_lateral --events events.bin --hop-window 3600 --runs 6 \
 *             --out lateral.bin
 *        ./auth_lateral --events events.bin --cpu-only
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
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>

#include "auth_common.cuh"

/* ------------------------- command-line parameters ------------------------ */
static const char  *events_path = "events.bin";
static const char  *out_path    = NULL;
static long long    learn_secs  = -1;      /* -1: first 25% of the time span */
static unsigned int nthreads    = 256;
static unsigned int num_runs    = 6;
static int          cpu_only    = 0;
static int          skip_cpu    = 0;

/* ------------------- constant memory: detector parameters ----------------- */
struct LateralParams {
    unsigned int learn_end;    /* absolute time: history only before this */
    unsigned int hop_win;      /* seconds                                 */
    float        w_new;
    float        w_rare;
    float        w_hop;
};
__constant__ LateralParams c_lp;
static LateralParams h_lp = { 0u, 3600u, 1.0f, 1.0f, 2.0f };

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const int has = (i + 1 < argc);
        if      (strcmp(argv[i], "--events") == 0 && has)     events_path = argv[++i];
        else if (strcmp(argv[i], "--out") == 0 && has)        out_path = argv[++i];
        else if (strcmp(argv[i], "--learn") == 0 && has)      learn_secs = atoll(argv[++i]);
        else if (strcmp(argv[i], "--hop-window") == 0 && has) h_lp.hop_win = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--w-new") == 0 && has)      h_lp.w_new = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--w-rare") == 0 && has)     h_lp.w_rare = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--w-hop") == 0 && has)      h_lp.w_hop = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--threads") == 0 && has)    nthreads = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && has)       num_runs = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--cpu-only") == 0)          cpu_only = 1;
        else if (strcmp(argv[i], "--no-cpu") == 0)            skip_cpu = 1;
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--events P] [--out P] [--learn SEC] "
                            "[--hop-window SEC] [--w-new F] [--w-rare F] [--w-hop F] "
                            "[--threads N] [--runs N] [--cpu-only] [--no-cpu]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (!check_block_size(nthreads)) exit(EXIT_FAILURE);
    if (h_lp.hop_win == 0) { fprintf(stderr, "Error: --hop-window must be >= 1\n"); exit(EXIT_FAILURE); }
    if (num_runs == 0) num_runs = 1;
    if (cpu_only) skip_cpu = 0;
}

/* ------------------------------ shared math ------------------------------- */
__host__ __device__ __forceinline__
float lateral_formula(unsigned int is_new, unsigned int pop, unsigned int hop,
                      const LateralParams &p)
{
    if (!is_new) return 0.0f;
    return p.w_new + p.w_rare / (1.0f + (float)pop) + (hop ? p.w_hop : 0.0f);
}

/* hop test */
__host__ __device__ __forceinline__
unsigned int hop_lookup(const unsigned long long *dict, unsigned int nd,
                        const unsigned long long *arr, unsigned int na,
                        unsigned int user, unsigned int src, unsigned int t,
                        unsigned int win)
{
    const unsigned long long q = ((unsigned long long)user << 32) | src;
    const unsigned int eid = find_dev(dict, nd, q);
    if (eid == nd) return 0u;                       /* user never landed on src */
    const unsigned long long base = (unsigned long long)eid << 32;
    const unsigned int lo = lower_bound_dev(arr, na, base | window_start(t, win));
    const unsigned int hi = upper_bound_dev(arr, na, base | t);
    return hi > lo ? 1u : 0u;
}

struct is_zero_u8 {
    __host__ __device__ bool operator()(unsigned char x) const { return x == 0; }
};
struct is_nonzero_u32 {
    __host__ __device__ bool operator()(unsigned int x) const { return x != 0; }
};
struct before_time {
    unsigned int t;
    __host__ __device__ bool operator()(unsigned int x) const { return x < t; }
};
struct hi32 {
    __host__ __device__ unsigned int operator()(unsigned long long x) const {
        return (unsigned int)(x >> 32);
    }
};

/* ------------------------------ GPU kernels ------------------------------- */
/* compacted successful events: k -> original index sidx[k] */
__global__ void gather_success(const unsigned int *sidx, const unsigned int *time,
                               const unsigned int *user, const unsigned int *src,
                               const unsigned int *dst, unsigned int *ct,
                               unsigned int *cu, unsigned int *cs, unsigned int *cd,
                               unsigned long long *ud, unsigned long long *du,
                               const unsigned int m)
{
    const unsigned int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k < m) {
        const unsigned int i = sidx[k];
        ct[k] = time[i]; cu[k] = user[i]; cs[k] = src[i]; cd[k] = dst[i];
        ud[k] = ((unsigned long long)user[i] << 32) | dst[i];   /* edge          */
        du[k] = ((unsigned long long)dst[i] << 32) | user[i];   /* for popularity */
    }
}

__global__ void edge_time_keys(const unsigned long long *ud, const unsigned int *ct,
                               const unsigned long long *dict, const unsigned int nd,
                               unsigned long long *key, unsigned int *perm,
                               const unsigned int m)
{
    const unsigned int k = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (k < m) {
        const unsigned int eid = lower_bound_dev(dict, nd, ud[k]);
        key[k]  = ((unsigned long long)eid << 32) | ct[k];
        perm[k] = k;
    }
}

__global__ void flag_new_edges(const unsigned long long *key, const unsigned int *perm,
                               const unsigned int *cs, const unsigned int *cd,
                               unsigned int *is_new, const unsigned int m)
{
    const unsigned int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (j < m) {
        const unsigned long long kj = key[j];
        const unsigned int first = lower_bound_dev(key, m, kj & 0xFFFFFFFF00000000ull);
        const unsigned int t  = (unsigned int)kj;
        const unsigned int t0 = (unsigned int)key[first];   /* first use of the edge */
        const unsigned int k  = perm[j];
        is_new[j] = (t == t0 && t >= c_lp.learn_end && cs[k] != cd[k]) ? 1u : 0u;
    }
}

__global__ void lateral_score(const unsigned long long *key, const unsigned int *perm,
                              const unsigned int *is_new, const unsigned int *sidx,
                              const unsigned int *cu, const unsigned int *cs,
                              const unsigned int *cd,
                              const unsigned long long *dict, const unsigned int nd,
                              const unsigned long long *arr, const unsigned int na,
                              const unsigned int *pop_ids, const unsigned int *pop_cnt,
                              const unsigned int np,
                              float *score_out, unsigned int *counters,
                              unsigned int *max_bits, const unsigned int m)
{
    /* shared memory: per-block new-edge count, hop count, max score */
    extern __shared__ unsigned int smem[];
    unsigned int *s_new = smem;
    unsigned int *s_hop = smem + blockDim.x;
    unsigned int *s_max = smem + 2 * blockDim.x;

    const unsigned int j = (blockIdx.x * blockDim.x) + threadIdx.x;
    unsigned int nw = 0, hp = 0;
    float s = 0.0f;
    if (j < m && is_new[j]) {
        const unsigned int k = perm[j];
        const unsigned int t = (unsigned int)key[j];
        hp = hop_lookup(dict, nd, arr, na, cu[k], cs[k], t, c_lp.hop_win);
        const unsigned int pi  = find_dev(pop_ids, np, cd[k]);
        const unsigned int pop = (pi < np) ? pop_cnt[pi] : 0u;
        s  = lateral_formula(1u, pop, hp, c_lp);
        nw = 1u;
        score_out[sidx[k]] = s;            /* others stay 0 from the memset */
    }
    s_new[threadIdx.x] = nw;
    s_hop[threadIdx.x] = hp;
    s_max[threadIdx.x] = __float_as_uint(s);
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_new[threadIdx.x] += s_new[threadIdx.x + stride];
            s_hop[threadIdx.x] += s_hop[threadIdx.x + stride];
            s_max[threadIdx.x]  = max(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        if (s_new[0]) atomicAdd(&counters[0], s_new[0]);
        if (s_hop[0]) atomicAdd(&counters[1], s_hop[0]);
        atomicMax(max_bits, s_max[0]);
    }
}

/* ------------------------------- GPU driver ------------------------------- */
struct Gpu {
    unsigned int *d_time, *d_user, *d_src, *d_dst;
    unsigned char *d_fail;
    unsigned int *d_sidx, *d_ct, *d_cu, *d_cs, *d_cd, *d_perm, *d_new;
    unsigned long long *d_ud, *d_du, *d_dict, *d_learn, *d_key, *d_arr;
    unsigned int *d_popdst, *d_pop_ids, *d_pop_cnt, *d_cnt, *d_max;
    float *d_score;
    unsigned int *h_small;                 /* pinned: new, hop, max bits */
    cudaStream_t s0, s1, s2;
    cudaEvent_t ev_start, ev_gathered, ev_dict, ev_pop, ev_stop;
};

static void gpu_setup(Gpu &g, unsigned int n)
{
    const size_t w = (size_t)n * sizeof(unsigned int), w8 = (size_t)n * 8;
    CUDA_CHECK(cudaMalloc(&g.d_time, w)); CUDA_CHECK(cudaMalloc(&g.d_user, w));
    CUDA_CHECK(cudaMalloc(&g.d_src, w));  CUDA_CHECK(cudaMalloc(&g.d_dst, w));
    CUDA_CHECK(cudaMalloc(&g.d_fail, n));
    CUDA_CHECK(cudaMalloc(&g.d_sidx, w)); CUDA_CHECK(cudaMalloc(&g.d_ct, w));
    CUDA_CHECK(cudaMalloc(&g.d_cu, w));   CUDA_CHECK(cudaMalloc(&g.d_cs, w));
    CUDA_CHECK(cudaMalloc(&g.d_cd, w));   CUDA_CHECK(cudaMalloc(&g.d_perm, w));
    CUDA_CHECK(cudaMalloc(&g.d_new, w));
    CUDA_CHECK(cudaMalloc(&g.d_ud, w8));  CUDA_CHECK(cudaMalloc(&g.d_du, w8));
    CUDA_CHECK(cudaMalloc(&g.d_dict, w8)); CUDA_CHECK(cudaMalloc(&g.d_learn, w8));
    CUDA_CHECK(cudaMalloc(&g.d_key, w8)); CUDA_CHECK(cudaMalloc(&g.d_arr, w8));
    CUDA_CHECK(cudaMalloc(&g.d_popdst, w)); CUDA_CHECK(cudaMalloc(&g.d_pop_ids, w));
    CUDA_CHECK(cudaMalloc(&g.d_pop_cnt, w));
    CUDA_CHECK(cudaMalloc(&g.d_cnt, 2 * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&g.d_max, sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&g.d_score, (size_t)n * sizeof(float)));
    CUDA_CHECK(cudaHostAlloc((void **)&g.h_small, 3 * sizeof(unsigned int), cudaHostAllocDefault));
    CUDA_CHECK(cudaStreamCreate(&g.s0)); CUDA_CHECK(cudaStreamCreate(&g.s1));
    CUDA_CHECK(cudaStreamCreate(&g.s2));
    CUDA_CHECK(cudaEventCreate(&g.ev_start)); CUDA_CHECK(cudaEventCreate(&g.ev_gathered));
    CUDA_CHECK(cudaEventCreate(&g.ev_dict));  CUDA_CHECK(cudaEventCreate(&g.ev_pop));
    CUDA_CHECK(cudaEventCreateWithFlags(&g.ev_stop, cudaEventBlockingSync));
}

static void gpu_teardown(Gpu &g)
{
    void *bufs[] = { g.d_time, g.d_user, g.d_src, g.d_dst, g.d_fail, g.d_sidx, g.d_ct,
                     g.d_cu, g.d_cs, g.d_cd, g.d_perm, g.d_new, g.d_ud, g.d_du, g.d_dict,
                     g.d_learn, g.d_key, g.d_arr, g.d_popdst, g.d_pop_ids, g.d_pop_cnt,
                     g.d_cnt, g.d_max, g.d_score };
    for (size_t i = 0; i < sizeof(bufs) / sizeof(bufs[0]); i++) cudaFree(bufs[i]);
    cudaFreeHost(g.h_small);
    cudaStreamDestroy(g.s0); cudaStreamDestroy(g.s1); cudaStreamDestroy(g.s2);
    cudaEventDestroy(g.ev_start); cudaEventDestroy(g.ev_gathered);
    cudaEventDestroy(g.ev_dict); cudaEventDestroy(g.ev_pop); cudaEventDestroy(g.ev_stop);
}

struct Stats { unsigned int new_edges, hops; float max_score; };

static float gpu_run(const Events &ev, float *h_out, Gpu &g, Stats *st)
{
    using thrust::device_ptr;
    const unsigned int n = ev.n;
    const size_t w = (size_t)n * sizeof(unsigned int);

    CUDA_CHECK(cudaEventRecord(g.ev_start, g.s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_time, ev.time, w, cudaMemcpyHostToDevice, g.s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_user, ev.user, w, cudaMemcpyHostToDevice, g.s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_src,  ev.src,  w, cudaMemcpyHostToDevice, g.s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_dst,  ev.dst,  w, cudaMemcpyHostToDevice, g.s0));
    CUDA_CHECK(cudaMemcpyAsync(g.d_fail, ev.fail, n, cudaMemcpyHostToDevice, g.s0));
    CUDA_CHECK(cudaMemsetAsync(g.d_score, 0, (size_t)n * sizeof(float), g.s0));
    CUDA_CHECK(cudaMemsetAsync(g.d_cnt, 0, 2 * sizeof(unsigned int), g.s0));
    CUDA_CHECK(cudaMemsetAsync(g.d_max, 0, sizeof(unsigned int), g.s0));

    /* successful logons only */
    device_ptr<unsigned int> sidx(g.d_sidx);
    const unsigned int m = (unsigned int)(thrust::copy_if(thrust::cuda::par.on(g.s0),
        thrust::counting_iterator<unsigned int>(0), thrust::counting_iterator<unsigned int>(n),
        device_ptr<unsigned char>(g.d_fail), sidx, is_zero_u8()) - sidx);

    unsigned int nd = 0, np = 0, na = 0;
    if (m > 0) {
        const unsigned int bm = (m + nthreads - 1) / nthreads;
        gather_success<<<bm, nthreads, 0, g.s0>>>(g.d_sidx, g.d_time, g.d_user, g.d_src,
            g.d_dst, g.d_ct, g.d_cu, g.d_cs, g.d_cd, g.d_ud, g.d_du, m);
        CUDA_CHECK(cudaEventRecord(g.ev_gathered, g.s0));

        /* stream1: user->host edge dictionary */
        CUDA_CHECK(cudaStreamWaitEvent(g.s1, g.ev_gathered, 0));
        device_ptr<unsigned long long> dict(g.d_dict);
        CUDA_CHECK(cudaMemcpyAsync(g.d_dict, g.d_ud, (size_t)m * 8, cudaMemcpyDeviceToDevice, g.s1));
        thrust::sort(thrust::cuda::par.on(g.s1), dict, dict + m);
        nd = (unsigned int)(thrust::unique(thrust::cuda::par.on(g.s1), dict, dict + m) - dict);
        CUDA_CHECK(cudaEventRecord(g.ev_dict, g.s1));

        /* stream2: distinct users per host during the learning period */
        CUDA_CHECK(cudaStreamWaitEvent(g.s2, g.ev_gathered, 0));
        device_ptr<unsigned long long> learn(g.d_learn);
        before_time pred; pred.t = h_lp.learn_end;
        const unsigned int nl = (unsigned int)(thrust::copy_if(thrust::cuda::par.on(g.s2),
            device_ptr<unsigned long long>(g.d_du), device_ptr<unsigned long long>(g.d_du) + m,
            device_ptr<unsigned int>(g.d_ct), learn, pred) - learn);
        thrust::sort(thrust::cuda::par.on(g.s2), learn, learn + nl);
        const unsigned int nu = (unsigned int)(thrust::unique(thrust::cuda::par.on(g.s2),
                                                              learn, learn + nl) - learn);
        device_ptr<unsigned int> popdst(g.d_popdst), pop_ids(g.d_pop_ids), pop_cnt(g.d_pop_cnt);
        thrust::transform(thrust::cuda::par.on(g.s2), learn, learn + nu, popdst, hi32());
        np = (unsigned int)(thrust::reduce_by_key(thrust::cuda::par.on(g.s2), popdst,
            popdst + nu, thrust::constant_iterator<unsigned int>(1), pop_ids, pop_cnt).first
            - pop_ids);
        CUDA_CHECK(cudaEventRecord(g.ev_pop, g.s2));

        /* stream0: per-edge time ordering, new edges, arrivals */
        CUDA_CHECK(cudaStreamWaitEvent(g.s0, g.ev_dict, 0));
        edge_time_keys<<<bm, nthreads, 0, g.s0>>>(g.d_ud, g.d_ct, g.d_dict, nd,
                                                  g.d_key, g.d_perm, m);
        device_ptr<unsigned long long> key(g.d_key), arr(g.d_arr);
        thrust::sort_by_key(thrust::cuda::par.on(g.s0), key, key + m,
                            device_ptr<unsigned int>(g.d_perm));
        flag_new_edges<<<bm, nthreads, 0, g.s0>>>(g.d_key, g.d_perm, g.d_cs, g.d_cd,
                                                  g.d_new, m);
        na = (unsigned int)(thrust::copy_if(thrust::cuda::par.on(g.s0), key, key + m,
            device_ptr<unsigned int>(g.d_new), arr, is_nonzero_u32()) - arr);

        CUDA_CHECK(cudaStreamWaitEvent(g.s0, g.ev_pop, 0));
        lateral_score<<<bm, nthreads, 3 * nthreads * sizeof(unsigned int), g.s0>>>(
            g.d_key, g.d_perm, g.d_new, g.d_sidx, g.d_cu, g.d_cs, g.d_cd,
            g.d_dict, nd, g.d_arr, na, g.d_pop_ids, g.d_pop_cnt, np,
            g.d_score, g.d_cnt, g.d_max, m);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpyAsync(h_out, g.d_score, (size_t)n * sizeof(float),
                               cudaMemcpyDeviceToHost, g.s0));
    CUDA_CHECK(cudaMemcpyAsync(g.h_small, g.d_cnt, 2 * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, g.s0));
    CUDA_CHECK(cudaMemcpyAsync(g.h_small + 2, g.d_max, sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, g.s0));
    CUDA_CHECK(cudaEventRecord(g.ev_stop, g.s0));
    CUDA_CHECK(cudaEventSynchronize(g.ev_stop));

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, g.ev_start, g.ev_stop);
    st->new_edges = g.h_small[0];
    st->hops      = g.h_small[1];
    memcpy(&st->max_score, &g.h_small[2], sizeof(float));
    return ms;
}

/* ------------------------------- CPU analogue ----------------------------- */
static double cpu_run(const Events &ev, float *out, Stats *st)
{
    auto t0 = std::chrono::steady_clock::now();
    const unsigned int n = ev.n;
    std::vector<unsigned int> sidx;
    sidx.reserve(n);
    for (unsigned int i = 0; i < n; i++)
        if (!ev.fail[i]) sidx.push_back(i);
    const unsigned int m = (unsigned int)sidx.size();

    /* edge dictionary and learning-period host popularity */
    std::vector<unsigned long long> dict(m), learn;
    learn.reserve(m);
    for (unsigned int k = 0; k < m; k++) {
        const unsigned int i = sidx[k];
        dict[k] = pack64(ev.user[i], ev.dst[i]);
        if (ev.time[i] < h_lp.learn_end) learn.push_back(pack64(ev.dst[i], ev.user[i]));
    }
    std::sort(dict.begin(), dict.end());
    dict.erase(std::unique(dict.begin(), dict.end()), dict.end());
    std::sort(learn.begin(), learn.end());
    learn.erase(std::unique(learn.begin(), learn.end()), learn.end());
    std::vector<unsigned int> pop_ids, pop_cnt;
    for (size_t q = 0; q < learn.size(); q++) {
        const unsigned int d = (unsigned int)(learn[q] >> 32);
        if (pop_ids.empty() || pop_ids.back() != d) { pop_ids.push_back(d); pop_cnt.push_back(0); }
        pop_cnt.back()++;
    }
    const unsigned int nd = (unsigned int)dict.size(), np = (unsigned int)pop_ids.size();

    /* (edge id, time) ordering and new-edge flags */
    std::vector<std::pair<unsigned long long, unsigned int> > kp(m);
    for (unsigned int k = 0; k < m; k++) {
        const unsigned int i = sidx[k];
        const unsigned int eid = lower_bound_dev(dict.data(), nd, pack64(ev.user[i], ev.dst[i]));
        kp[k] = std::make_pair(pack64(eid, ev.time[i]), k);
    }
    std::sort(kp.begin(), kp.end());
    std::vector<unsigned long long> key(m), arr;
    std::vector<unsigned char> is_new(m);
    for (unsigned int j = 0; j < m; j++) key[j] = kp[j].first;
    unsigned int group_start = 0;          /* first index of the current edge */
    for (unsigned int j = 0; j < m; j++) {
        if (j == 0 || (key[j] >> 32) != (key[j - 1] >> 32)) group_start = j;
        const unsigned int i = sidx[kp[j].second];
        const unsigned int t = (unsigned int)key[j];
        is_new[j] = (t == (unsigned int)key[group_start] && t >= h_lp.learn_end &&
                     ev.src[i] != ev.dst[i]) ? 1 : 0;
        if (is_new[j]) arr.push_back(key[j]);
    }
    const unsigned int na = (unsigned int)arr.size();

    memset(out, 0, (size_t)n * sizeof(float));
    unsigned int n_new = 0, n_hop = 0;
    float best = 0.0f;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 4096) reduction(+:n_new, n_hop) reduction(max:best)
#endif
    for (long long jj = 0; jj < (long long)m; jj++) {
        const unsigned int j = (unsigned int)jj;
        if (!is_new[j]) continue;
        const unsigned int i = sidx[kp[j].second];
        const unsigned int hp = hop_lookup(dict.data(), nd, arr.data(), na, ev.user[i],
                                           ev.src[i], (unsigned int)key[j], h_lp.hop_win);
        const unsigned int pi  = find_dev(pop_ids.data(), np, ev.dst[i]);
        const unsigned int pop = (pi < np) ? pop_cnt[pi] : 0u;
        const float s = lateral_formula(1u, pop, hp, h_lp);
        out[i] = s;
        n_new++; n_hop += hp;
        if (s > best) best = s;
    }
    auto t1 = std::chrono::steady_clock::now();
    st->new_edges = n_new; st->hops = n_hop; st->max_score = best;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

/* main */
int main(int argc, char **argv)
{
    parse_args(argc, argv);
    const int use_gpu = !cpu_only;

    printf("=== Authentication threat detection: lateral-movement detector ===\n");
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
    const unsigned long long span = (unsigned long long)ev.tmax - ev.tmin;
    const unsigned long long learn = (learn_secs < 0) ? span / 4 : (unsigned long long)learn_secs;
    h_lp.learn_end = (unsigned int)std::min<unsigned long long>(ev.tmin + learn, 0xFFFFFFFFull);
    printf("CPU threads: %d | hop window: %us | learning: t < %u (%llus) | runs: %u\n",
           cpu_threads, h_lp.hop_win, h_lp.learn_end, learn, num_runs);
    printf("Loaded %u events from %s, t = %u..%u\n\n", n, events_path, ev.tmin, ev.tmax);

    float *gpu_out = use_gpu ? (float *)host_alloc((size_t)n * sizeof(float), 1) : NULL;
    float *cpu_out = !skip_cpu ? (float *)malloc((size_t)n * sizeof(float)) : NULL;
    Stats gs = {}, cs = {};
    double g_tot = 0, g_best = 0, c_tot = 0, c_best = 0;

    if (use_gpu) {
        CUDA_CHECK(cudaMemcpyToSymbol(c_lp, &h_lp, sizeof(LateralParams)));
        Gpu g;
        gpu_setup(g, n);
        gpu_run(ev, gpu_out, g, &gs);                                  /* warmup */
        for (unsigned int r = 0; r < num_runs; r++) {
            const double t = gpu_run(ev, gpu_out, g, &gs);
            g_tot += t;
            if (r == 0 || t < g_best) g_best = t;
        }
        gpu_teardown(g);
        g_tot /= num_runs;
        printf("GPU (avg of %u): total %.3f ms | best %.3f ms\n", num_runs, g_tot, g_best);
    }
    if (!skip_cpu) {
        for (unsigned int r = 0; r < num_runs; r++) {
            const double t = cpu_run(ev, cpu_out, &cs);
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
        match = (max_diff <= 1e-5 && gs.new_edges == cs.new_edges && gs.hops == cs.hops);
        printf("Verification: GPU vs CPU max |diff| = %.3e, new edges %u vs %u, "
               "hops %u vs %u -> %s\n", max_diff, gs.new_edges, cs.new_edges,
               gs.hops, cs.hops, match ? "MATCH" : "MISMATCH");
        printf("Speedup (avg end-to-end incl. transfers): %.2fx\n", c_tot / g_tot);
    }

    const Stats &S = use_gpu ? gs : cs;
    const float *final_out = use_gpu ? gpu_out : cpu_out;
    printf("\nNew user->host edges: %u (%.3f%%) | hop chains: %u | max score %.3f\n",
           S.new_edges, 100.0 * S.new_edges / n, S.hops, S.max_score);
    if (out_path) {
        write_scores_f32(out_path, final_out, n);
        printf("Wrote per-event lateral-movement scores to %s\n", out_path);
    }

    printf("RESULT {\"detector\":\"lateral\",\"n\":%u,\"gpu_ms\":%.4f,\"gpu_best_ms\":%.4f,"
           "\"cpu_ms\":%.4f,\"cpu_best_ms\":%.4f,\"cpu_threads\":%d,\"new_edges\":%u,"
           "\"hops\":%u,\"learn_end\":%u,\"max_abs_diff\":%.6g,\"match\":%d}\n",
           n, use_gpu ? g_tot : -1.0, use_gpu ? g_best : -1.0,
           skip_cpu ? -1.0 : c_tot, skip_cpu ? -1.0 : c_best, cpu_threads,
           S.new_edges, S.hops, h_lp.learn_end, max_diff, match);

    host_free(gpu_out, 1);
    free(cpu_out);
    free_events(ev, use_gpu);
    if (use_gpu) cudaDeviceReset();
    return (match == 0) ? EXIT_FAILURE : EXIT_SUCCESS;
}
