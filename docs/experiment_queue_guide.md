# Running long experiment queues safely (tmux + logs + resume + locks)

A reusable recipe for running many long jobs (training runs, evaluations, sweeps) on a shared
Linux/GPU server so that:

- every job writes its own **log file**, and the queue keeps a short **summary log**;
- the jobs **keep running** when you close the terminal, VSCode, SSH or your laptop;
- after a crash or an accidental Ctrl-C you **re-run the same command** and it **continues** where it stopped;
- you can run **several queues in parallel** (e.g. one per GPU) without two of them doing the same job;
- a typo or missing file **stops the queue in seconds** instead of failing every job one by one.

It needs nothing beyond `bash` and `tmux`.

---

## 1. Quick start

```bash
# 1. Copy the template (section 3) to your project as run_queue.sh, edit the JOBS section
chmod +x run_queue.sh

# 2. Start it inside tmux, so it survives closing the terminal
tmux new -s exp                                   # new session called "exp"
cd /path/to/project
mkdir -p logs && ./run_queue.sh 2>&1 | tee -a logs/queue.log

# 3. Detach (leave it running):  Ctrl-b  then  d
# 4. Come back later:
tmux attach -t exp
cat logs/queue.log                                # one line per job: start / done / FAIL
tail -f logs/<job_name>.log                       # follow one job live
```

To use 2 GPUs, open a second tmux window (`Ctrl-b c`) a minute later and start the **same command**
again. The locks make the two copies share the work.

---

## 2. How it works

### 2.1 Logs: redirection and `tee`

Inside the script each job runs as

```bash
python train.py ... > "logs/$name.log" 2>&1
```

- `>` writes everything the job prints into its own file.
- `2>&1` sends errors (stderr) to the same file, so a crash's traceback is saved too.
- `export PYTHONUNBUFFERED=1` makes Python write each line immediately; without it, output is
  buffered and `tail -f` shows nothing for minutes.

The queue itself only prints short `start / done / FAIL` lines. Starting it with
`./run_queue.sh 2>&1 | tee -a logs/queue.log` shows those lines on screen **and** appends them to
`queue.log` (`-a` = append, so a restart does not erase history).

Tip: progress bars (tqdm) write `\r` updates, so a log can contain very long lines. To read only
the epoch lines: `tr '\r' '\n' < logs/job.log | grep epoch`.

### 2.2 Surviving a closed terminal: tmux

tmux runs your shell inside a server process on the machine. Your terminal is only a window onto
it. Closing VSCode, losing SSH, or shutting your laptop does **not** stop anything inside tmux.

You can also start a queue already detached, without attaching first:

```bash
tmux new-session -d -s exp -n lane1 "cd $PWD && ./run_queue.sh 2>&1 | tee -a logs/queue.log; exec bash"
tmux new-window  -t exp    -n lane2 "cd $PWD && ./run_queue.sh 2>&1 | tee -a logs/queue.log; exec bash"
```

(`; exec bash` keeps the window open after the queue ends, so you can read the last lines.)

**What tmux does not protect against:** pressing Ctrl-C *inside* the attached window, `tmux kill-session`,
or the server rebooting. That is what sections 2.3–2.4 are for.

### 2.3 Continuing after a stop: a "done" marker

Each job must write one file **as its very last step**, for example `results/<name>/metrics.json`.
The script treats that file as proof the job finished:

```bash
[[ -f "$OUT/$name/metrics.json" ]] && return     # finished earlier: skip
```

Re-running the queue skips every finished job and starts the first unfinished one.

**Limitation:** a job interrupted halfway restarts from the beginning, unless your training code
can resume from its own checkpoint (see section 6). You lose at most one job's progress.

**Requirement on your code:** write the marker file *last*, after everything else is saved. If it is
written early, a crash later would make the queue think the job is complete.

### 2.4 Parallel queues without collisions: lock folders

Before starting a job, a queue tries to claim it:

```bash
mkdir "$LOCKS/$name" 2>/dev/null || return       # someone else has it: skip
```

`mkdir` is **atomic**: if two queues try at the same moment, exactly one succeeds. So you can start
the same script N times (N GPUs) and each job runs exactly once.

When a job **finishes**, its lock is removed (the done-marker now protects it), so deleting a job's
results folder is enough to make it run again. If a job **fails**, the lock is also removed so a later
run can retry it. Only if you **Ctrl-C** a job (or the server reboots) does the lock stay behind, and the
job will be skipped until you remove it by hand (section 5).

### 2.5 Stopping fast on setup errors

If a job fails within the first few minutes, it is almost always a setup error (typo, wrong path,
missing package, GPU out of memory). The script then **stops the whole queue** and prints the end
of the job's log, instead of failing all remaining jobs in a few seconds and leaving you a wall of
`FAIL` lines the next morning. A failure *after* that window is logged and the queue moves on.

### 2.6 Picking a GPU (optional)

On a shared server, pick the GPU with the most free memory just before each job starts and pin
the job to it with `CUDA_VISIBLE_DEVICES`. The template has a small `pick_gpu` function that uses
`nvidia-smi`. If the server is busy, it waits and retries.

---

## 3. The template

Copy this into your project as `run_queue.sh` and edit only the **CONFIG** and **JOBS** sections.

```bash
#!/usr/bin/env bash
# Queue of long jobs: per-job logs, resumable, parallel-safe (run several copies, e.g. one per GPU),
# stops early on setup errors. Start inside tmux:
#   mkdir -p logs && ./run_queue.sh 2>&1 | tee -a logs/queue.log
set -u
cd "$(dirname "$0")"

# ---------------- CONFIG ----------------
PY=venv/bin/python               # the project's Python
OUT=results                      # each job writes to $OUT/<name>/
DONE_FILE=metrics.json           # written by the job as its LAST step = "job finished"
LOGS=logs
LOCKS=$OUT/.locks
SETUP_ERROR_SECONDS=180          # a failure faster than this stops the queue
MIN_FREE_MIB=12000               # GPU must have this much free memory (0 = don't pick a GPU)
# ----------------------------------------

export PYTHONUNBUFFERED=1
mkdir -p "$LOGS" "$LOCKS" "$OUT"

pick_gpu() {   # prints the index of the GPU with most free memory (>= MIN_FREE_MIB); waits if none
  while true; do
    local best
    best=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
           | awk -F', ' -v min="$MIN_FREE_MIB" '$2 >= min {print $2, $1}' | sort -nr | head -1 | cut -d' ' -f2)
    [[ -n $best ]] && { echo "$best"; return; }
    echo "[$(date '+%F %T')] no GPU with ${MIN_FREE_MIB} MiB free, waiting 60 s" >&2
    sleep 60
  done
}

run() {   # run <name> <command...>
  local name=$1; shift
  [[ -f "$OUT/$name/$DONE_FILE" ]] && return              # finished earlier
  mkdir "$LOCKS/$name" 2>/dev/null || return              # claimed by another copy of this queue
  local gpu="" t0=$SECONDS
  if (( MIN_FREE_MIB > 0 )); then gpu=$(pick_gpu); fi
  echo "[$(date '+%F %T')] start $name ${gpu:+(GPU $gpu)}  log: $LOGS/$name.log"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu "$@" > "$LOGS/$name.log" 2>&1
  local status=$? elapsed=$(( SECONDS - t0 ))
  if [[ $status -eq 0 && -f "$OUT/$name/$DONE_FILE" ]]; then
    echo "[$(date '+%F %T')] done  $name  ($(( elapsed / 60 )) min)"
    rmdir "$LOCKS/$name"                                  # the done-marker now protects it
  else
    echo "[$(date '+%F %T')] FAIL  $name (exit $status) - see $LOGS/$name.log"
    rmdir "$LOCKS/$name"                                  # let a later run retry it
    if (( elapsed < SETUP_ERROR_SECONDS )); then
      echo "[stop] failed after ${elapsed}s - probably a setup error. Last lines:"
      tail -20 "$LOGS/$name.log"
      exit 1
    fi
  fi
}

# ---------------- JOBS ----------------
# One `run` line per job: a unique name, then the command. The command must write
# $OUT/<name>/$DONE_FILE as its last step. Loops are fine:
for lr in 0.001 0.0003; do
  for seed in 0 1 2; do
    run "lr${lr}_s${seed}" $PY train.py --lr "$lr" --seed "$seed" --out "$OUT/lr${lr}_s${seed}"
  done
done
# ---------------------------------------
echo "[$(date '+%F %T')] queue finished"
```

---

## 4. tmux cheat sheet

| Action | Command / keys |
|---|---|
| New named session | `tmux new -s exp` |
| New session, detached, running a command | `tmux new-session -d -s exp -n lane1 "cmd; exec bash"` |
| Detach (leave running) | `Ctrl-b` then `d` |
| List sessions | `tmux ls` |
| Re-attach | `tmux attach -t exp` |
| New window / next window / window N | `Ctrl-b c` / `Ctrl-b n` / `Ctrl-b 0..9` |
| List windows | `tmux list-windows -t exp` |
| Scroll back (then `q` to quit) | `Ctrl-b [` |
| Kill a session (stops everything in it) | `tmux kill-session -t exp` |

---

## 5. When something stopped: recovery checklist

```bash
cat logs/queue.log | grep -E "FAIL|stop"         # what failed
tail -50 logs/<job>.log                          # why

ls results/.locks/                               # claimed jobs
# A lock with no results/<job>/metrics.json and no process running = stale. Check what is running:
pgrep -af train.py
rmdir results/.locks/<job>                       # free a stale lock

tmux attach -t exp                               # restart the queue in its window
./run_queue.sh 2>&1 | tee -a logs/queue.log      # finished jobs are skipped automatically
```

To stop a queue **gently** (let the current job finish, start no new ones), find the queue's
`bash` process and kill only that; the running Python job continues and finishes normally:

```bash
pgrep -af run_queue.sh        # note the PID of "bash ./run_queue.sh"
kill <PID>
```

---

## 6. Pitfalls

- **Never edit a bash script while it is running.** Bash reads scripts piece by piece as it runs, so
  editing the file can make the running queue execute garbage. Copy it to a new name, or wait until it ends.
- **Don't start a second copy of a queue that has no locks.** Both copies would start the same job
  and write to the same folder.
- **Write the done-marker last.** Otherwise a crash after the marker is written looks like success.
- **Unique job names.** The name is the folder, the log and the lock: two jobs with the same name collide.
- **Buffered output.** Without `PYTHONUNBUFFERED=1` (or `python -u`) the logs look empty while jobs run.
- **Shared GPUs.** Another user's job can take the memory between the GPU check and your job starting.
  A failure from that hits the setup-error rule and stops the queue: read the log, then restart.
- **Resuming inside a job** (not just between jobs) needs support in your training code: save
  `model + optimizer + scheduler + epoch` every epoch to `last.pt`, and on start, load it if it
  exists. Then an interrupted job resumes from its last epoch instead of epoch 1.

---

## 7. Alternatives

| Tool | When to prefer it |
|---|---|
| `nohup ./run_queue.sh > logs/queue.log 2>&1 &` | No tmux available. Survives logout but you cannot re-attach to watch it live. |
| `screen` | Older servers without tmux; same idea (`screen -S exp`, detach `Ctrl-a d`, `screen -r exp`). |
| SLURM / PBS (`sbatch`) | University clusters with a scheduler: use its job arrays instead of this script. |
| W&B / MLflow | Add on top for experiment dashboards; they do not replace the queue. |
