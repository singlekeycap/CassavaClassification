#!/usr/bin/env python3
"""Pick the best free GPU automatically, then launch a command on it.

Usage (from the project root):
    python gpu_run.py -- python -m Deep_Learning.train --model resnet50 --imbalance sampler
    python gpu_run.py --min-free-gb 20 --wait -- python -m Deep_Learning.train ...
    python gpu_run.py --dry-run          # just show the GPU table and the pick

How a GPU is chosen:
  1. Cards whose name contains "Display" or that have < --min-total-gb memory are skipped,
     as are any indices passed with --exclude.
  2. A card must have at least --min-free-gb free memory.
  3. Among the remaining cards, the one with the lowest average utilization (sampled
     --samples times) wins; ties are broken by most free memory, then fewest processes.

The chosen GPU is pinned by UUID in CUDA_VISIBLE_DEVICES, so inside the launched
program it is simply `cuda` / `cuda:0`. torch is never imported here.
"""
import argparse
import os
import subprocess
import sys
import time

QUERY = "index,uuid,name,memory.used,memory.total,utilization.gpu"


def smi(*args):
    return subprocess.run(["nvidia-smi", *args], check=True, capture_output=True, text=True).stdout


def read_gpus(samples, interval):
    """Return a list of dicts, with utilization averaged over `samples` readings."""
    gpus = {}
    for i in range(samples):
        for line in smi(f"--query-gpu={QUERY}", "--format=csv,noheader,nounits").strip().splitlines():
            idx, uuid, name, used, total, util = [x.strip() for x in line.split(",")]
            g = gpus.setdefault(uuid, dict(index=int(idx), uuid=uuid, name=name, util=[]))
            g["used"], g["total"] = float(used), float(total)          # MiB, latest reading
            g["util"].append(float(util) if util.replace(".", "").isdigit() else 100.0)
        if i < samples - 1:
            time.sleep(interval)

    procs = {}
    try:
        for line in smi("--query-compute-apps=gpu_uuid", "--format=csv,noheader").strip().splitlines():
            procs[line.strip()] = procs.get(line.strip(), 0) + 1
    except subprocess.CalledProcessError:
        pass

    out = []
    for g in gpus.values():
        g["util"] = sum(g["util"]) / len(g["util"])
        g["free"] = g["total"] - g["used"]
        g["procs"] = procs.get(g["uuid"], 0)
        out.append(g)
    return sorted(out, key=lambda g: g["index"])


def why_skipped(g, args):
    if g["index"] in args.exclude:
        return "excluded"
    if "display" in g["name"].lower() or g["total"] < args.min_total_gb * 1024:
        return "not a compute card"
    if g["free"] < args.min_free_gb * 1024:
        return f"< {args.min_free_gb:g} GB free"
    if args.max_util is not None and g["util"] > args.max_util:
        return f"util > {args.max_util:g}%"
    return None


def pick(gpus, args):
    ok = [g for g in gpus if why_skipped(g, args) is None]
    ok.sort(key=lambda g: (round(g["util"]), -g["free"], g["procs"]))
    return ok[: args.num_gpus] if len(ok) >= args.num_gpus else []


def show(gpus, chosen, args):
    chosen_ids = {g["uuid"] for g in chosen}
    print(f"{'':2}{'idx':>3}  {'name':<24}{'free GB':>9}{'total GB':>10}{'util%':>7}{'procs':>7}  note")
    for g in gpus:
        mark = "->" if g["uuid"] in chosen_ids else ""
        note = "CHOSEN" if mark else (why_skipped(g, args) or "")
        print(f"{mark:2}{g['index']:>3}  {g['name'][:23]:<24}{g['free']/1024:>9.1f}{g['total']/1024:>10.1f}"
              f"{g['util']:>7.0f}{g['procs']:>7}  {note}")


def auto_select_gpu(min_free_gb=12, min_total_gb=16, wait=True, poll=60, samples=5, interval=0.5):
    """Call this at the very top of a script, BEFORE `import torch`.

    Picks the best GPU and sets CUDA_VISIBLE_DEVICES so `cuda` means that card.
    Does nothing when CUDA_VISIBLE_DEVICES is already set (you chose a GPU yourself),
    when nvidia-smi is unavailable, or when there is only one compute card to choose from.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[gpu_run] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} already set; not changing it")
        return
    args = argparse.Namespace(min_free_gb=min_free_gb, min_total_gb=min_total_gb, max_util=None,
                              num_gpus=1, exclude=[])
    try:
        while True:
            gpus = read_gpus(samples, interval)
            compute = [g for g in gpus if why_skipped(g, args) != "not a compute card"]
            if len(compute) <= 1:
                return  # nothing to choose between (e.g. a single-GPU laptop)
            chosen = pick(gpus, args)
            show(gpus, chosen, args)
            if chosen:
                break
            if not wait:
                print("[gpu_run] no GPU has enough free memory; leaving GPU choice to CUDA")
                return
            print(f"[gpu_run] no GPU meets the requirements; retrying in {poll:g}s ...\n", flush=True)
            time.sleep(poll)
    except (OSError, subprocess.CalledProcessError):
        print("[gpu_run] nvidia-smi not available; skipping GPU auto-selection")
        return
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = chosen[0]["uuid"]
    print(f"[gpu_run] using GPU {chosen[0]['index']} ({chosen[0]['uuid']})\n", flush=True)


def main():
    argv = sys.argv[1:]
    cmd = []
    if "--" in argv:
        cut = argv.index("--")
        argv, cmd = argv[:cut], argv[cut + 1:]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-free-gb", type=float, default=12, help="minimum free memory on the GPU (default 12)")
    ap.add_argument("--min-total-gb", type=float, default=16, help="ignore cards smaller than this (default 16)")
    ap.add_argument("--max-util", type=float, default=None, help="optionally skip GPUs busier than this %%")
    ap.add_argument("--num-gpus", type=int, default=1, help="how many GPUs to give the job (default 1)")
    ap.add_argument("--exclude", type=int, nargs="*", default=[], help="nvidia-smi indices never to use")
    ap.add_argument("--samples", type=int, default=5, help="utilization readings to average (default 5)")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between readings (default 0.5)")
    ap.add_argument("--wait", action="store_true", help="if no GPU qualifies, keep polling until one does")
    ap.add_argument("--poll", type=float, default=60, help="seconds between polls with --wait (default 60)")
    ap.add_argument("--dry-run", action="store_true", help="show the table and the pick, launch nothing")
    args = ap.parse_args(argv)

    if not cmd and not args.dry_run:
        ap.error("give the command to run after `--`, e.g.  -- python -m Deep_Learning.train")

    while True:
        gpus = read_gpus(args.samples, args.interval)
        chosen = pick(gpus, args)
        show(gpus, chosen, args)
        if chosen or not args.wait or args.dry_run:
            break
        print(f"[gpu_run] no GPU meets the requirements; retrying in {args.poll:g}s ...\n", flush=True)
        time.sleep(args.poll)

    if not chosen:
        sys.exit("[gpu_run] no suitable GPU found (try --wait, or lower --min-free-gb)")
    if args.dry_run:
        return

    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
               CUDA_VISIBLE_DEVICES=",".join(g["uuid"] for g in chosen))
    print(f"[gpu_run] using GPU {','.join(str(g['index']) for g in chosen)} "
          f"({env['CUDA_VISIBLE_DEVICES']})\n[gpu_run] $ {' '.join(cmd)}\n", flush=True)
    os.execvpe(cmd[0], cmd, env)   # replaces this process: Ctrl-C and exit code pass straight through


if __name__ == "__main__":
    main()
