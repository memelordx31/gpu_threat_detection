#!/usr/bin/env python3
"""
run_pipeline.py
GPU-Accelerated Authentication Threat Detection project -- main driver.

Steps
  1. Preprocess: cut a slice of LANL cyber1 auth.txt (plain or .gz), write it
     as text (auth_slice.txt) and as events.bin (integer ids, timestamps,
     fail flags), and label every event that appears in redteam.txt
     (time, user, src computer, dst computer).
  2. Build the CUDA programs (nvcc) and, with --legacy, the CPU baseline (g++).
  3. Run the three detectors: auth_lateral (lateral movement), auth_fanout
     (fan-out), auth_failburst (failed-logon bursts), then auth_score, which
     combines them into one per-event score. Each program times its GPU
     pipeline and its own CPU analogue and checks that they agree.
  4. Optional (--legacy): the two original hash-based modules
     (auth_scan_memtypes, auth_streams_libs) and their CPU analogue
     (cpu_baseline), for their memory-type / stream timing studies.
  5. Evaluate: speedup per stage and end to end against the target, and
     detection quality (ROC AUC, average precision, recall at the alert
     budget and at 1%) of each detector and of the combined score against
     the red-team labels.

Logs go to <work>/logs/, the summary to <work>/results.json and the 50
highest-scoring events to <work>/top_events.csv.

  python3 run_pipeline.py --start-time auto              # real LANL data in ../
  python3 run_pipeline.py --auth data/auth.txt --redteam data/redteam.txt --cpu-only
"""

import argparse
import gzip
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EVT_MAGIC = 0x31545645   # "EVT1"
SCR_MAGIC = 0x52435344   # "DSCR"


# --------------------------------------------------------------------------- #
# arguments
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    io = p.add_argument_group("data")
    io.add_argument("--auth", default=str(HERE.parent / "auth.txt"),
                    help="LANL cyber1 auth.txt or auth.txt.gz (default: ../auth.txt)")
    io.add_argument("--redteam", default=str(HERE.parent / "redteam.txt"),
                    help="LANL cyber1 redteam.txt or .gz (default: ../redteam.txt)")
    io.add_argument("--events", type=int, default=4 << 20,
                    help="number of auth events in the slice (default 4194304)")
    io.add_argument("--start-line", type=int, default=0,
                    help="skip this many auth lines before the slice")
    io.add_argument("--start-time", default=None,
                    help="start the slice at the first event with time >= T; 'auto' = "
                         "2 hours before the first red-team event")
    io.add_argument("--work", default=str(HERE / "work"), help="output directory")

    b = p.add_argument_group("build / run")
    b.add_argument("--arch", default="sm_75", help="nvcc -arch (default sm_75)")
    b.add_argument("--build", default=str(HERE / "build"), help="binary directory")
    b.add_argument("--no-build", action="store_true", help="reuse existing binaries")
    b.add_argument("--cpu-only", action="store_true",
                   help="no GPU: run only the CPU analogues (checks the pipeline)")
    b.add_argument("--legacy", action="store_true",
                   help="also run auth_scan_memtypes / auth_streams_libs / cpu_baseline")
    b.add_argument("--threads", type=int, default=256, help="CUDA threads per block")
    b.add_argument("--runs", type=int, default=6, help="timed repetitions per stage")
    b.add_argument("--cpu-threads", type=int, default=None,
                   help="OMP_NUM_THREADS for the CPU analogues (default: all cores)")

    d = p.add_argument_group("detectors")
    d.add_argument("--learn", type=int, default=-1,
                   help="lateral: seconds of history before anything counts as new "
                        "(default: first 25%% of the slice)")
    d.add_argument("--hop-window", type=int, default=3600,
                   help="lateral: max seconds between arriving on a host and hopping on")
    d.add_argument("--fan-key", default="src", choices=["src", "user"],
                   help="fan-out: entity whose distinct destinations are counted")
    d.add_argument("--fan-short", type=int, default=600, help="fan-out short bucket, s")
    d.add_argument("--fan-long", type=int, default=3600, help="fan-out long bucket, s")
    d.add_argument("--fan-alert", type=int, default=10,
                   help="fan-out: distinct destinations per short bucket that alert")
    d.add_argument("--burst-key", default="user", choices=["user", "src", "dst"],
                   help="burst: entity whose failed logons are counted")
    d.add_argument("--burst-short", type=int, default=60, help="burst short window, s")
    d.add_argument("--burst-long", type=int, default=3600, help="burst long window, s")
    d.add_argument("--burst-alert", type=int, default=5,
                   help="burst: failures in the short window that alert")
    d.add_argument("--weights", default="1,1,1",
                   help="combined-score weights for lateral,fanout,burst")
    d.add_argument("--alert", type=float, default=0.95,
                   help="combined-score alert threshold, in [0, 1)")
    d.add_argument("--mc", type=int, default=16384, help="legacy streams MC replicates")
    d.add_argument("--sample", type=int, default=512, help="legacy streams MC draws")
    d.add_argument("--target-speedup", type=float, default=3.0)
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 1. preprocessing
# --------------------------------------------------------------------------- #
def open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", newline="")
    return open(path, "r", newline="")


def load_redteam(path):
    """Set of (time, user, src, dst) tuples, plus the earliest red-team time."""
    events, first = set(), None
    with open_text(path) as fh:
        for line in fh:
            f = line.rstrip("\r\n").split(",")
            if len(f) < 4:
                continue
            t = int(f[0])
            events.add((t, f[1], f[2], f[3]))
            first = t if first is None else min(first, t)
    return events, first


def preprocess(args, work):
    print("[1/5] preprocessing", flush=True)
    t0 = time.time()
    redteam, first_rt = load_redteam(args.redteam)
    # the C/CUDA readers need plain text: keep an uncompressed copy in work/
    rt_plain = work / "redteam.txt"
    with open_text(args.redteam) as src, open(rt_plain, "w", newline="") as dst:
        shutil.copyfileobj(src, dst)
    print(f"      redteam: {len(redteam)} events, first at t={first_rt}")

    start_time = None
    if args.start_time is not None:
        if args.start_time == "auto":
            start_time = max(0, (first_rt or 0) - 7200)
        else:
            start_time = int(args.start_time)

    slice_path = work / "auth_slice.txt"
    n_target = args.events
    times = np.empty(n_target, dtype=np.uint32)
    users = np.empty(n_target, dtype=np.uint32)
    srcs = np.empty(n_target, dtype=np.uint32)
    dsts = np.empty(n_target, dtype=np.uint32)
    fails = np.empty(n_target, dtype=np.uint8)
    labels = np.empty(n_target, dtype=np.uint8)
    user_id, comp_id = {}, {}

    n = 0
    skipped = 0
    with open_text(args.auth) as fh, open(slice_path, "w", newline="") as out:
        for line in fh:
            if skipped < args.start_line:
                skipped += 1
                continue
            f = line.rstrip("\r\n").split(",")
            if len(f) < 9:
                continue                     # malformed: drop it from every detector
            t = int(f[0])
            if start_time is not None and t < start_time:
                skipped += 1
                continue
            if not line.endswith("\n"):
                line += "\n"
            if len(line) >= 1023:            # CUDA modules read with fgets(1024)
                continue
            out.write(line)
            times[n] = t
            users[n] = user_id.setdefault(f[1], len(user_id))
            srcs[n] = comp_id.setdefault(f[3], len(comp_id))
            dsts[n] = comp_id.setdefault(f[4], len(comp_id))
            fails[n] = 1 if f[8].strip().lower().startswith("fail") else 0
            labels[n] = 1 if (t, f[1], f[3], f[4]) in redteam else 0
            n += 1
            if n == n_target:
                break
    if n == 0:
        sys.exit("error: the slice is empty (check --auth / --start-line / --start-time)")

    times, users, srcs, dsts, fails, labels = (a[:n] for a in
                                               (times, users, srcs, dsts, fails, labels))
    with open(work / "events.bin", "wb") as fh:
        np.array([EVT_MAGIC, 1, n], dtype=np.uint32).tofile(fh)
        for a in (times, users, srcs, dsts, fails):
            a.tofile(fh)
    np.save(work / "labels.npy", labels)

    info = {
        "events": int(n), "skipped_lines": int(skipped),
        "time_range": [int(times[0]), int(times[-1])],
        "users": len(user_id), "computers": len(comp_id),
        "failed_logons": int(fails.sum()), "redteam_events": int(labels.sum()),
        "seconds": round(time.time() - t0, 2),
    }
    print(f"      slice: {n} events, t={info['time_range'][0]}..{info['time_range'][1]}, "
          f"{info['failed_logons']} failures, {info['redteam_events']} red-team events "
          f"({info['seconds']} s)")
    if info["redteam_events"] == 0:
        print("      WARNING: no red-team events in this slice, so detection metrics "
              "are undefined. Try --start-time auto and/or a larger --events.")
    return slice_path, rt_plain, labels, info


# --------------------------------------------------------------------------- #
# 2. build
# --------------------------------------------------------------------------- #
DETECTORS = ["auth_lateral", "auth_fanout", "auth_failburst", "auth_score"]
LEGACY = ["auth_scan_memtypes", "auth_streams_libs"]


def build(args, bdir):
    print("[2/5] building", flush=True)
    bdir.mkdir(parents=True, exist_ok=True)
    nv = ["nvcc", "-O2", f"-arch={args.arch}", f"-I{HERE}"]
    omp = ["-Xcompiler", "-fopenmp", "-lgomp"]
    cmds = [nv + [str(HERE / f"{d}.cu"), "-o", str(bdir / d)] + omp for d in DETECTORS]
    if args.legacy:
        cmds.append(["g++", "-O2", "-fopenmp", str(HERE / "cpu_baseline.cpp"),
                     "-o", str(bdir / "cpu_baseline")])
        if not args.cpu_only:
            cmds.append(nv + [str(HERE / "auth_scan_memtypes.cu"), "-o",
                              str(bdir / "auth_scan_memtypes")])
            cmds.append(nv + [str(HERE / "auth_streams_libs.cu"), "-o",
                              str(bdir / "auth_streams_libs"), "-lcurand"])
    for c in cmds:
        if shutil.which(c[0]) is None:
            sys.exit(f"error: {c[0]} not found on PATH")
        print("      " + " ".join(c))
        r = subprocess.run(c, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"build failed:\n{r.stdout}{r.stderr}")


# --------------------------------------------------------------------------- #
# 3. run stages
# --------------------------------------------------------------------------- #
def run_stage(name, cmd, logdir, env):
    print(f"      $ {' '.join(map(str, cmd))}", flush=True)
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True, env=env)
    (logdir / f"{name}.log").write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        sys.exit(f"stage '{name}' failed (exit {r.returncode}); "
                 f"see {logdir / (name + '.log')}\n{r.stderr[-2000:]}")
    result = None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
    if result is None:
        sys.exit(f"stage '{name}' printed no RESULT line; see its log")
    print(f"        done in {time.time() - t0:.1f} s", flush=True)
    return result


def read_scores(path):
    with open(path, "rb") as fh:
        magic, dtype, n = np.fromfile(fh, dtype=np.uint32, count=3)
        if magic != SCR_MAGIC:
            sys.exit(f"error: {path} is not a DSCR score file")
        arr = np.fromfile(fh, dtype=np.uint32 if dtype == 0 else np.float32, count=n)
    if arr.size != n:
        sys.exit(f"error: {path} is truncated")
    return arr


# --------------------------------------------------------------------------- #
# 5. evaluation metrics
# --------------------------------------------------------------------------- #
def roc_auc(scores, labels):
    """Mann-Whitney AUC with average ranks for ties."""
    labels = labels.astype(bool)
    pos, neg = int(labels.sum()), int((~labels).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    s = scores.astype(np.float64)
    order = np.argsort(s, kind="mergesort")
    _, first, counts = np.unique(s[order], return_index=True, return_counts=True)
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.repeat(first + (counts + 1) / 2.0, counts)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def average_precision(scores, labels):
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return float("nan")
    order = np.argsort(-scores.astype(np.float64), kind="mergesort")
    hits = labels[order]
    cum = np.cumsum(hits)
    prec = cum / np.arange(1, len(hits) + 1)
    return float(prec[hits].sum() / pos)


def recall_at(scores, labels, k):
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0 or k <= 0:
        return float("nan")
    top = np.argsort(-scores.astype(np.float64), kind="mergesort")[:k]
    return float(labels[top].sum() / pos)


def fmt(x, nd=4):
    return "  n/a " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    work = Path(args.work)
    bdir = Path(args.build)
    logdir = work / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if args.cpu_threads:
        env["OMP_NUM_THREADS"] = str(args.cpu_threads)

    slice_path, rt_plain, labels, data_info = preprocess(args, work)
    n = data_info["events"]
    if not args.no_build:
        build(args, bdir)

    print("[3/5] running detectors", flush=True)
    T, R = args.threads, args.runs
    cpu_flag = ["--cpu-only"] if args.cpu_only else []
    events = work / "events.bin"
    out = {k: work / f"{k}.bin" for k in ("lateral", "fanout", "burst", "combined")}
    res = {}

    res["lateral"] = run_stage("lateral", [
        bdir / "auth_lateral", "--events", events, "--learn", args.learn,
        "--hop-window", args.hop_window, "--threads", T, "--runs", R,
        "--out", out["lateral"]] + cpu_flag, logdir, env)
    res["fanout"] = run_stage("fanout", [
        bdir / "auth_fanout", "--events", events, "--key", args.fan_key,
        "--short", args.fan_short, "--long", args.fan_long, "--alert", args.fan_alert,
        "--threads", T, "--runs", R, "--out", out["fanout"]] + cpu_flag, logdir, env)
    res["burst"] = run_stage("failburst", [
        bdir / "auth_failburst", "--events", events, "--key", args.burst_key,
        "--short", args.burst_short, "--long", args.burst_long,
        "--alert", args.burst_alert, "--threads", T, "--runs", R,
        "--out", out["burst"]] + cpu_flag, logdir, env)

    print("[4/5] combined scoring", flush=True)
    res["score"] = run_stage("score", [
        bdir / "auth_score", "--det1", out["lateral"], "--det2", out["fanout"],
        "--det3", out["burst"], "--weights", args.weights, "--alert", args.alert,
        "--threads", T, "--runs", R, "--out", out["combined"]] + cpu_flag, logdir, env)

    channels = {
        "lateral": read_scores(out["lateral"]),
        "fanout": read_scores(out["fanout"]),
        "failburst": read_scores(out["burst"]),
        "combined": read_scores(out["combined"]),
    }
    for k, v in channels.items():
        if v.size != n:
            sys.exit(f"error: {k} has {v.size} scores, expected {n}")

    # ----------------------- optional legacy modules ---------------------- #
    legacy = None
    if args.legacy:
        print("      legacy hash-based modules", flush=True)
        size_b = 64 * 1024
        res["cpu_baseline"] = run_stage("legacy_cpu_baseline", [
            bdir / "cpu_baseline", "--fileA", slice_path, "--fileB", rt_plain,
            "--elements", n, "--sizeB", size_b, "--runs", max(1, min(R, 3)),
            "--out1", work / "cpu_memtypes.bin", "--out2", work / "cpu_streams.bin"],
            logdir, env)
        if not args.cpu_only:
            res["memtypes"] = run_stage("legacy_memtypes", [
                bdir / "auth_scan_memtypes", "--file", slice_path, "--elements", n,
                "--threads", T, "--blocks", (n + T - 1) // T, "--runs", R,
                "--out", work / "memtypes.bin"], logdir, env)
            res["streams"] = run_stage("legacy_streams", [
                bdir / "auth_streams_libs", "--fileA", slice_path, "--fileB", rt_plain,
                "--sizeA", n, "--sizeB", size_b, "--threads", T,
                "--blocks", (max(n, size_b) + T - 1) // T, "--mc", args.mc,
                "--sample", args.sample, "--out", work / "streams.bin"], logdir, env)
            cb = res["cpu_baseline"]
            legacy = {
                "memtypes": {"gpu_ms": res["memtypes"]["gmem_ms"], "cpu_ms": cb["det1_ms"],
                             "gpu_eq_cpu": bool(np.array_equal(
                                 read_scores(work / "memtypes.bin"),
                                 read_scores(work / "cpu_memtypes.bin")))},
                "streams": {"gpu_ms": res["streams"]["concurrent_ms"], "cpu_ms": cb["det2_ms"],
                            "gpu_eq_cpu": bool(np.array_equal(
                                read_scores(work / "streams.bin"),
                                read_scores(work / "cpu_streams.bin")))},
            }
        channels["legacy memtypes"] = read_scores(work / ("cpu_memtypes.bin" if args.cpu_only
                                                          else "memtypes.bin"))
        channels["legacy streams"] = read_scores(work / ("cpu_streams.bin" if args.cpu_only
                                                         else "streams.bin"))

    # --------------------------- verification ----------------------------- #
    verify = {}
    if not args.cpu_only:
        for k in ("lateral", "fanout", "burst", "score"):
            verify[f"{k}_gpu_vs_cpu"] = bool(res[k]["match"] == 1)
        if legacy:
            verify["legacy_memtypes_gpu_eq_cpu"] = legacy["memtypes"]["gpu_eq_cpu"]
            verify["legacy_streams_gpu_eq_cpu"] = legacy["streams"]["gpu_eq_cpu"]
        for k, ok in verify.items():
            print(f"      {k:28s} {'OK' if ok else 'FAILED'}")

    # ------------------------------ speedup ------------------------------- #
    print("[5/5] evaluation", flush=True)
    speed = None
    if not args.cpu_only:
        rows = [(name, res[k]["gpu_ms"], res[k]["cpu_ms"]) for name, k in
                (("lateral", "lateral"), ("fanout", "fanout"),
                 ("failburst", "burst"), ("combined score", "score"))]
        gpu_tot = sum(r[1] for r in rows)
        cpu_tot = sum(r[2] for r in rows)
        speed = {
            "cpu_threads": res["lateral"]["cpu_threads"],
            "stages": [{"stage": s, "gpu_ms": g, "cpu_ms": c, "speedup": c / g}
                       for s, g, c in rows],
            "total_gpu_ms": gpu_tot, "total_cpu_ms": cpu_tot,
            "total_speedup": cpu_tot / gpu_tot,
            "target": args.target_speedup,
            "meets_target": cpu_tot / gpu_tot >= args.target_speedup,
            "legacy": legacy,
        }
        print(f"\n  Speedup vs CPU analogue ({speed['cpu_threads']} CPU threads; "
              f"GPU times include host<->device transfers)")
        print(f"  {'stage':16s} {'GPU ms':>10s} {'CPU ms':>11s} {'speedup':>9s}")
        for st in speed["stages"]:
            print(f"  {st['stage']:16s} {st['gpu_ms']:10.3f} {st['cpu_ms']:11.3f} "
                  f"{st['speedup']:8.2f}x")
        print(f"  {'TOTAL':16s} {gpu_tot:10.3f} {cpu_tot:11.3f} {cpu_tot / gpu_tot:8.2f}x"
              f"   target {args.target_speedup:.1f}x -> "
              f"{'MET' if speed['meets_target'] else 'NOT MET'}")
        if legacy:
            for k, v in legacy.items():
                print(f"  (legacy {k}: GPU {v['gpu_ms']:.3f} ms, CPU {v['cpu_ms']:.3f} ms, "
                      f"{v['cpu_ms'] / v['gpu_ms']:.2f}x)")

    # ------------------------- detection quality -------------------------- #
    pos = int(labels.sum())
    budget = int(res["score"]["alerts"])
    detection = {}
    print(f"\n  Detection vs red-team labels ({pos} positives of {n} events; "
          f"k = {budget} = combined alerts)")
    print(f"  {'channel':16s} {'ROC AUC':>8s} {'avg prec':>9s} {'recall@k':>9s} "
          f"{'recall@1%':>10s}")
    for name, s in channels.items():
        m = {"roc_auc": roc_auc(s, labels), "avg_precision": average_precision(s, labels),
             "recall_at_k": recall_at(s, labels, budget),
             "recall_at_1pct": recall_at(s, labels, max(1, n // 100))}
        detection[name] = m
        print(f"  {name:16s} {fmt(m['roc_auc']):>8s} {fmt(m['avg_precision']):>9s} "
              f"{fmt(m['recall_at_k']):>9s} {fmt(m['recall_at_1pct']):>10s}")

    # top-scoring events with their raw log lines, for inspection
    comb = channels["combined"]
    top = np.argsort(-comb, kind="mergesort")[:50]
    wanted = set(int(i) for i in top)
    raw = {}
    with open(slice_path) as fh:
        for i, line in enumerate(fh):
            if i in wanted:
                raw[i] = line.rstrip("\r\n")
    with open(work / "top_events.csv", "w") as fh:
        fh.write("rank,index,combined,lateral,fanout,failburst,redteam,raw\n")
        for r, i in enumerate(top, 1):
            fh.write(f"{r},{i},{comb[i]:.6f},{channels['lateral'][i]:.4f},"
                     f"{channels['fanout'][i]:.4f},{channels['failburst'][i]:.4f},"
                     f"{int(labels[i])},\"{raw[int(i)]}\"\n")

    summary = {"data": data_info, "args": vars(args), "stages": res,
               "verification": verify, "speedup": speed, "detection": detection}
    (work / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n  wrote {work / 'results.json'} and {work / 'top_events.csv'}")
    if verify and not all(verify.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
