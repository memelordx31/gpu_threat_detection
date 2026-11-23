/* ============================================================================
 * auth_common.cuh
 * GPU-Accelerated Authentication Threat Detection project
 * Helpers shared by the detector components (auth_lateral.cu, auth_fanout.cu,
 * auth_failburst.cu): error checking, events.bin / DSCR file I/O, pinned host
 * allocation and host+device binary searches.
 *
 * events.bin (written by run_pipeline.py), all little-endian:
 *   uint32 magic 'EVT1', uint32 version (1), uint32 n,
 *   uint32 time[n], uint32 user[n], uint32 src[n], uint32 dst[n], uint8 fail[n]
 *
 * DSCR score file (written by every detector, read by auth_score.cu):
 *   uint32 magic 'DSCR', uint32 dtype (0 = uint32, 1 = float32), uint32 n, data
 * ==========================================================================*/
#ifndef AUTH_COMMON_CUH
#define AUTH_COMMON_CUH

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EVT_MAGIC 0x31545645u   /* "EVT1" */
#define SCR_MAGIC 0x52435344u   /* "DSCR" */
#define DTYPE_U32 0u
#define DTYPE_F32 1u

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

/* ----------------------------- host memory -------------------------------- */
static void *host_alloc(size_t bytes, int pinned)
{
    void *p = NULL;
    if (bytes == 0) bytes = 1;
    if (pinned)
        CUDA_CHECK(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
    else
        p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "host allocation of %zu bytes failed\n", bytes);
        exit(EXIT_FAILURE);
    }
    return p;
}

static void host_free(void *p, int pinned)
{
    if (!p) return;
    if (pinned) cudaFreeHost(p); else free(p);
}

/* ------------------------------ events.bin -------------------------------- */
struct Events {
    unsigned int   n;
    unsigned int   tmin, tmax;
    unsigned int  *time;
    unsigned int  *user;     /* source user id      */
    unsigned int  *src;      /* source computer id  */
    unsigned int  *dst;      /* destination comp id */
    unsigned char *fail;     /* 1 = failed logon    */
};

static void read_exact(FILE *fp, void *dst, size_t bytes, const char *path)
{
    if (fread(dst, 1, bytes, fp) != bytes) {
        fprintf(stderr, "Error: '%s' is truncated\n", path);
        exit(EXIT_FAILURE);
    }
}

static Events read_events(const char *path, int pinned)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'. Create it with run_pipeline.py "
                        "(preprocessing step).\n", path);
        exit(EXIT_FAILURE);
    }
    unsigned int hdr[3];
    read_exact(fp, hdr, sizeof(hdr), path);
    if (hdr[0] != EVT_MAGIC || hdr[1] != 1u || hdr[2] == 0u) {
        fprintf(stderr, "Error: '%s' is not a version-1 events file\n", path);
        exit(EXIT_FAILURE);
    }
    Events ev;
    ev.n = hdr[2];
    const size_t w = (size_t)ev.n * sizeof(unsigned int);
    ev.time = (unsigned int *)host_alloc(w, pinned);
    ev.user = (unsigned int *)host_alloc(w, pinned);
    ev.src  = (unsigned int *)host_alloc(w, pinned);
    ev.dst  = (unsigned int *)host_alloc(w, pinned);
    ev.fail = (unsigned char *)host_alloc(ev.n, pinned);
    read_exact(fp, ev.time, w, path);
    read_exact(fp, ev.user, w, path);
    read_exact(fp, ev.src,  w, path);
    read_exact(fp, ev.dst,  w, path);
    read_exact(fp, ev.fail, ev.n, path);
    fclose(fp);
    ev.tmin = ev.tmax = ev.time[0];
    for (unsigned int i = 1; i < ev.n; i++) {
        if (ev.time[i] < ev.tmin) ev.tmin = ev.time[i];
        if (ev.time[i] > ev.tmax) ev.tmax = ev.time[i];
    }
    return ev;
}

static void free_events(Events &ev, int pinned)
{
    host_free(ev.time, pinned); host_free(ev.user, pinned);
    host_free(ev.src,  pinned); host_free(ev.dst,  pinned);
    host_free(ev.fail, pinned);
}

static void write_scores_f32(const char *path, const float *s, unsigned int n)
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

/* --------------------- binary searches (host + device) -------------------- */
/* first index with a[idx] >= v */
template <typename T>
__host__ __device__ __forceinline__
unsigned int lower_bound_dev(const T *a, unsigned int n, T v)
{
    unsigned int lo = 0, hi = n;
    while (lo < hi) {
        const unsigned int mid = lo + ((hi - lo) >> 1);
        if (a[mid] < v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* first index with a[idx] > v */
template <typename T>
__host__ __device__ __forceinline__
unsigned int upper_bound_dev(const T *a, unsigned int n, T v)
{
    unsigned int lo = 0, hi = n;
    while (lo < hi) {
        const unsigned int mid = lo + ((hi - lo) >> 1);
        if (a[mid] <= v) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* index of v in sorted a, or n if absent */
template <typename T>
__host__ __device__ __forceinline__
unsigned int find_dev(const T *a, unsigned int n, T v)
{
    const unsigned int i = lower_bound_dev(a, n, v);
    return (i < n && a[i] == v) ? i : n;
}

/* first time stamp of the trailing window (t - win, t] */
__host__ __device__ __forceinline__
unsigned int window_start(unsigned int t, unsigned int win)
{
    return (t + 1u > win) ? (t + 1u - win) : 0u;
}

static inline unsigned long long pack64(unsigned int hi, unsigned int lo)
{
    return ((unsigned long long)hi << 32) | lo;
}

static int check_block_size(unsigned int nthreads)
{
    if (nthreads == 0 || nthreads > 1024 || (nthreads & (nthreads - 1)) != 0) {
        fprintf(stderr, "Error: --threads must be a power of two in [1, 1024]\n");
        return 0;
    }
    return 1;
}

#endif /* AUTH_COMMON_CUH */
