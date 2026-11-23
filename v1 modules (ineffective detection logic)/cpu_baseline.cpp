/* ============================================================================
 * cpu_baseline.cpp
 * GPU-Accelerated Authentication Threat Detection project
 * CPU analogues of auth_scan_memtypes.cu and auth_streams_libs.cu, for the
 * speedup evaluation. (auth_failburst.cu and auth_score.cu carry their own
 * CPU analogues.)
 *
 *   detector 1: the scan_gmem anomaly loop over every auth record. The masks
 *               are read through volatile globals, as scan_gmem reads them
 *               from __device__ globals, so the compiler cannot constant-fold
 *               the loop away (it can and does fold the literal version).
 *   detector 2: score_one() over dataset A (auth) and dataset B (redteam),
 *               i.e. the work of the two concurrent streams.
 *
 * Both use OpenMP when compiled with -fopenmp. Record parsing and hashing are
 * identical to the CUDA modules, so the --out files must match the GPU ones
 * bit for bit (run_pipeline.py checks this).
 *
 * Build: g++ -O2 -fopenmp cpu_baseline.cpp -o cpu_baseline
 * Run:   ./cpu_baseline --fileA ../auth.txt --fileB ../redteam.txt \
 *            --elements 1048576 --runs 3 --out1 cpu_det1.bin --out2 cpu_det2.bin
 * ==========================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#define KERNEL_LOOP 4096
#define SCORE_LOOP  4096

static const char  *pathA    = "../auth.txt";
static const char  *pathB    = "../redteam.txt";
static const char  *out1     = NULL;
static const char  *out2     = NULL;
static unsigned int elements = 1024 * 1024;
static unsigned int sizeB    = 64 * 1024;
static unsigned int num_runs = 3;

/* same values as the __device__ globals in auth_scan_memtypes.cu */
static volatile unsigned int g_breach_mask = 0x55555555u;
static volatile unsigned int g_ip_mask     = 0x77777777u;
static volatile unsigned int g_repeat_mask = 0x33333333u;
static volatile unsigned int g_admin_mask  = 0x11111111u;

static void parse_args(int argc, char **argv)
{
    for (int i = 1; i < argc; i++) {
        const int has = (i + 1 < argc);
        if      (strcmp(argv[i], "--fileA") == 0 && has)    pathA = argv[++i];
        else if (strcmp(argv[i], "--fileB") == 0 && has)    pathB = argv[++i];
        else if (strcmp(argv[i], "--elements") == 0 && has) elements = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--sizeB") == 0 && has)    sizeB = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--runs") == 0 && has)     num_runs = (unsigned)atoi(argv[++i]);
        else if (strcmp(argv[i], "--out1") == 0 && has)     out1 = argv[++i];
        else if (strcmp(argv[i], "--out2") == 0 && has)     out2 = argv[++i];
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            fprintf(stderr, "Usage: %s [--fileA P] [--fileB P] [--elements N] [--sizeB N] "
                            "[--runs N] [--out1 P] [--out2 P]\n", argv[0]);
            exit(EXIT_FAILURE);
        }
    }
    if (num_runs == 0) num_runs = 1;
}

/* ---------------- ingestion: copied from the two CUDA modules ------------- */
static unsigned int hash_auth_record(const char *line)          /* module 1 */
{
    unsigned int h = 2166136261u;
    int field = 0;
    for (const char *p = line; *p && *p != '\n' && *p != '\r'; p++) {
        if (*p == ',') { field++; h ^= ','; h *= 16777619u; continue; }
        if (field == 0) continue;
        h ^= (unsigned char)*p;
        h *= 16777619u;
    }
    return h;
}

static void hash_record(const char *line, int is_redteam,        /* module 2 */
                        unsigned int *user_hash, unsigned int *ip_hash)
{
    unsigned int hu = 2166136261u, hi = 2166136261u;
    int field = 0;
    for (const char *p = line; *p && *p != '\n' && *p != '\r'; p++) {
        if (*p == ',') {
            field++;
            hu ^= ','; hu *= 16777619u;
            hi ^= ','; hi *= 16777619u;
            continue;
        }
        if (field == 0) continue;
        unsigned int *target = (is_redteam ? (field == 1) : (field <= 2)) ? &hu : &hi;
        *target ^= (unsigned char)*p;
        *target *= 16777619u;
    }
    *user_hash = hu;
    *ip_hash   = hi;
}

struct Dataset { std::vector<unsigned int> h1, a, b; };

static Dataset read_file(const char *path, int is_redteam, unsigned int max_records)
{
    FILE *fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "Error: cannot open '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    Dataset d;
    char line[1024];
    while (d.a.size() < max_records && fgets(line, sizeof(line), fp)) {
        unsigned int a, b;
        hash_record(line, is_redteam, &a, &b);
        d.a.push_back(a);
        d.b.push_back(b);
        if (!is_redteam) d.h1.push_back(hash_auth_record(line));
    }
    fclose(fp);
    if (d.a.empty()) {
        fprintf(stderr, "Error: no records read from '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    return d;
}

static void write_u32(const char *path, const std::vector<unsigned int> &v)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Error: cannot write '%s'\n", path); exit(EXIT_FAILURE); }
    const unsigned int hdr[3] = { 0x52435344u, 0u, (unsigned int)v.size() };
    fwrite(hdr, sizeof(hdr), 1, fp);
    fwrite(v.data(), sizeof(unsigned int), v.size(), fp);
    fclose(fp);
}

/* ------------------------------ detector math ----------------------------- */
static void scan_cpu(const unsigned int *in, unsigned int *out, unsigned int n)
{
    const unsigned int m0 = g_breach_mask, m1 = g_ip_mask,
                       m2 = g_repeat_mask, m3 = g_admin_mask;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long t = 0; t < (long long)n; t++) {
        unsigned int d = in[t];
        const unsigned int attempt = in[t];
        for (int i = 0; i < KERNEL_LOOP; i++) {
            d ^= m0; d |= m1; d &= m2; d |= m3; d += attempt;
        }
        out[t] = d;
    }
}

static inline unsigned int score_one(unsigned int a, unsigned int b)
{
    unsigned int s = 0;
    for (int i = 0; i < SCORE_LOOP; i++) {
        unsigned int diff = (a * 2654435761u) ^ (b * 40503u);
        s += diff * diff;
        a = a * 1664525u + 1013904223u;
        b = b * 22695477u + 1u;
    }
    return s;
}

static void score_cpu(const unsigned int *a, const unsigned int *b,
                      unsigned int *s, unsigned int n)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (long long t = 0; t < (long long)n; t++)
        s[t] = score_one(a[t], b[t]);
}

template <typename F>
static double time_ms(F f, unsigned int runs, double *best)
{
    double tot = 0;
    for (unsigned int r = 0; r < runs; r++) {
        auto t0 = std::chrono::steady_clock::now();
        f();
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        tot += ms;
        if (r == 0 || ms < *best) *best = ms;
    }
    return tot / runs;
}

int main(int argc, char **argv)
{
    parse_args(argc, argv);
    int threads = 1;
#ifdef _OPENMP
    threads = omp_get_max_threads();
#endif
    printf("=== CPU analogue of detectors 1 & 2 (%d threads, %u runs) ===\n",
           threads, num_runs);

    Dataset A = read_file(pathA, 0, elements);
    Dataset B = read_file(pathB, 1, sizeB);
    const unsigned int nA = (unsigned int)A.a.size(), nB = (unsigned int)B.a.size();
    printf("Dataset A: %s -> %u records | Dataset B: %s -> %u records\n",
           pathA, nA, pathB, nB);

    std::vector<unsigned int> d1(nA), sA(nA), sB(nB);
    double b1 = 0, b2 = 0;
    const double t1 = time_ms([&] { scan_cpu(A.h1.data(), d1.data(), nA); }, num_runs, &b1);
    const double t2 = time_ms([&] {
        score_cpu(A.a.data(), A.b.data(), sA.data(), nA);
        score_cpu(B.a.data(), B.b.data(), sB.data(), nB);
    }, num_runs, &b2);

    printf("Detector 1 (memtypes scan): avg %.3f ms | best %.3f ms\n", t1, b1);
    printf("Detector 2 (A+B scoring):   avg %.3f ms | best %.3f ms\n", t2, b2);
    if (out1) write_u32(out1, d1);
    if (out2) write_u32(out2, sA);

    printf("RESULT {\"detector\":\"cpu_baseline\",\"n\":%u,\"nB\":%u,\"det1_ms\":%.4f,"
           "\"det1_best_ms\":%.4f,\"det2_ms\":%.4f,\"det2_best_ms\":%.4f,"
           "\"cpu_threads\":%d}\n", nA, nB, t1, b1, t2, b2, threads);
    return EXIT_SUCCESS;
}
