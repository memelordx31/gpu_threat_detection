# GPU-Accelerated Authentication Threat Detection

Three CUDA detectors score every LANL cyber1 (https://csr.lanl.gov/data/cyber1/) authentication event: lateral movement, fan-out and failed-logon bursts. A CUDA scoring kernel combines the three into one per-event score, and a Python driver handles preprocessing, the runs and the evaluation. Each CUDA program also carries a CPU version of the same algorithm. It times both versions and checks that they agree, which is what the 3x speedup target is measured against.

## Directory layout

The data files go in the **parent** directory of the code. That is where every module looks by default (`../auth.txt`, `../redteam.txt`).

```
gpu_threat_detection/                     
├── auth.txt          (or auth.txt.gz)      LANL cyber1 authentication log
├── redteam.txt       (or redteam.txt.gz)   LANL cyber1 red-team events
└── src/         
    ├── README.md
    ├── Makefile                  manual build (optional)
    ├── run_pipeline.py           main script: preprocess, build, run, evaluate
    │
    ├── auth_common.cuh           shared I/O + binary-search helpers
    ├── auth_lateral.cu           detector 1: lateral movement
    ├── auth_fanout.cu            detector 2: fan-out
    ├── auth_failburst.cu         detector 3: failed-logon bursts
    ├── auth_score.cu             combined per-event scoring kernel
    │
    ├── build/                    created: compiled binaries
    └── work/                     created: slice, events.bin, scores, logs, results
```

## Requirements

- An NVIDIA GPU and the CUDA toolkit (nvcc, Thrust and cuRAND come with it). The default target is `sm_75` (T4, RTX 20xx). Pass `--arch` for other GPUs: `sm_80` A100, `sm_86` RTX 30xx, `sm_89` RTX 40xx / L4, `sm_90` H100.
- g++ with OpenMP (the CPU analogues use every core by default).
- Python 3.8+ with numpy.

## Test run (about 5 minutes, no download)

```bash
python3 run_pipeline.py --arch sm_75          # use your GPU's arch
```

No GPU on the machine? `python3 run_pipeline.py --cpu-only` runs only the CPU versions and the evaluation. It checks the pipeline, not the speedup.

What to check in the output:

1. **Verification**: every line should end in `OK`, for example `lateral_gpu_vs_cpu  OK`. The run exits non-zero if any stage's GPU output differs from its CPU version.
2. **Speedup table**: GPU and CPU time for each stage, plus the total against the 3.0x target (`MET` / `NOT MET`). GPU times include the host-to-device and device-to-host copies.
3. **Detection table**: ROC AUC, average precision and recall for each detector and for the combined score. On the synthetic data (CPU-only run, which gives the same scores) these came out roughly as:

   | channel   | ROC AUC | recall@1% |
   |-----------|---------|-----------|
   | lateral   | 0.95    | 0.91      |
   | fanout    | 0.60    | 0.36      |
   | failburst | 0.86    | 0.05      |
   | combined  | 0.97    | 0.70      |

   Each detector covers one stage of an attack chain. Fan-out catches the compromise event from the host that just did reconnaissance (about a third of the labelled events). The burst detector covers the brute-force step, and lateral covers the hops. So no single detector reaches every red-team event, and the combined score is the best overall.

The outputs land in `work/`:

- `results.json`: every number above plus all settings
- `top_events.csv`: the 50 highest-scoring events, their raw log lines and their red-team labels
- `logs/*.log`: the full output of each program

## Run on the real LANL cyber1 data

1. Download `auth.txt.gz` and `redteam.txt.gz` from https://csr.lanl.gov/data/cyber1/ (it asks you to accept terms of use). Put them in `lanl_project/`. You don't need to decompress them: the driver reads `.gz` directly and writes the slice it uses as plain text into `work/`.
2. Run:

```bash
cd ~/gpu_threat_detection/src
python3 run_pipeline.py --auth ../auth.txt.gz --redteam ../redteam.txt.gz \
        --start-time auto --events 4194304 --arch sm_75
```

- `--start-time auto` starts the slice 2 hours before the first red-team event. Red-team activity does not begin at the start of the log, so a slice from line 0 has no positive labels.
- The LANL log runs at about 200 events/s, so 4 M events covers about 5–6 hours. Lateral-movement detection needs history: the first 25% of the slice is used only to learn who normally uses which host. For more history and more red-team events, use `--events 16777216`, or set `--learn SECONDS` explicitly.
- `--legacy` also runs the two original hash-based modules and their CPU baseline, so their timing studies appear in the same report.

## Running the pieces by hand

```bash
make ARCH=sm_75                                   # or: make legacy
python3 run_pipeline.py --no-build ...            # reuse build/

# every detector reads work/events.bin (written by the driver's preprocessing step)
build/auth_lateral   --events work/events.bin --hop-window 3600 --out work/lateral.bin
build/auth_fanout    --events work/events.bin --key src --short 600 --long 3600 --out work/fanout.bin
build/auth_failburst --events work/events.bin --key user --short 60 --long 3600 --out work/burst.bin
build/auth_score     --det1 work/lateral.bin --det2 work/fanout.bin --det3 work/burst.bin \
                     --weights 1,1,1 --alert 0.95 --out work/combined.bin
```

Every program accepts `--threads N` (a power of two), `--runs N`, `--cpu-only` and `--no-cpu`. Each prints one `RESULT {...}` JSON line, which the driver parses.

## What each detector computes

- **Lateral movement** (`auth_lateral.cu`), on successful logons only. An event is a *new edge* when it is the user's first logon to that destination after the learning period. It is a *hop* when the user itself arrived at the source host through a new edge within `--hop-window`. Score = `1 + 1/(1 + users seen on the destination while learning) + 2·hop`. GPU techniques: Thrust `copy_if` / `sort` / `unique` / `reduce_by_key`, three streams (the edge dictionary and host popularity are built in parallel), parameters in `__constant__`, and `__shared__` block reductions.
- **Fan-out** (`auth_fanout.cu`). Counts distinct destination computers per source host (or per user, with `--key user`) in 10-minute and 1-hour buckets, and compares each count with that host's own average. Score = `0.25·log2(1+D_S) + 0.25·log2(1+D_L) + Σ max(0, log2((1+D)/(1+mean)))`. The two scales run on two streams.
- **Failed-logon bursts** (`auth_failburst.cu`). Counts the user's failed logons in trailing 60-second and 1-hour windows; a successful logon right after a burst scores high. It uses a Thrust sort and scan, the two windows on two streams, and binary-search window counts.
- **Combined score** (`auth_score.cu`). Ranks each detector's scores on its own (the fraction of events with a lower score), then takes a weighted mean (`--weights`). Weights live in `__constant__`; the three channels are ranked on three streams; alert count, max score and a histogram are reduced in `__shared__` memory.

## Troubleshooting

- **`nvcc fatal: Unsupported gpu architecture`** or **`no kernel image is available`**: pass the right `--arch` for your GPU.
- **`no red-team events in this slice`**: use `--start-time auto` and/or a larger `--events`.
- **`entities x buckets does not fit 32 bits`** (fan-out): use a wider `--fan-short` or a shorter slice.
- **Out of GPU memory**: use a smaller `--events`. The lateral detector needs about 120 bytes per event, plus Thrust's temporary storage.
- **Speedup below target**: try more events. Fixed costs such as transfers and Thrust's temporary allocations dominate small slices. Also report `--cpu-threads`: a single-threaded CPU baseline inflates the speedup, and the all-cores default is the honest comparison.
