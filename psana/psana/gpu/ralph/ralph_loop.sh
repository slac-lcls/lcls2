#!/usr/bin/env bash
# Ralph loop driver — psana2 GPU Jungfrau calibration MVP.
#
# Each iteration: ensure a GPU holder allocation is RUNNING, then a fresh
# `claude -p` reads PROMPT.md, does ONE verified unit of work, journals it in
# PROGRESS.md, and commits. Stops on the `LOOP DONE` journal sentinel, on the
# iteration cap, or after two consecutive iterations that fail to append a
# journal entry (malfunction detector — a well-behaved iteration ALWAYS
# journals, even when blocked). A journal entry ending in `BLOCKED: ...`
# makes the driver wait before the next iteration instead of proceeding;
# if that line names a slurm job (`BLOCKED: job <ID> ...`), the driver polls
# squeue until the job leaves the queue, so waiting out a congested queue
# costs zero agent iterations.
#
# Run from anywhere, ideally inside tmux on an SDF login node:
#   tmux new -s ralph
#   psana/psana/gpu/ralph/ralph_loop.sh
set -uo pipefail

RALPH_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$RALPH_DIR/../../../.." && pwd)"
cd "$ROOT"

# Child claude processes inherit this environment (conda ps_20241122,
# PYTHONPATH -> install tree). Conda's activate/deactivate hooks reference
# unset variables, so nounset must be off while sourcing.
set +u
source "$ROOT/setup_env.sh"
set -u

MAX_ITERS="${MAX_ITERS:-25}"
MODEL="${MODEL:-opus}"
BLOCKED_SLEEP="${BLOCKED_SLEEP:-900}"     # wait after a BLOCKED: iteration (no job ID)
BLOCKED_WAIT_MAX="${BLOCKED_WAIT_MAX:-14400}"  # max wait for a BLOCKED: job <ID> to clear
ALLOC_WAIT_MAX="${ALLOC_WAIT_MAX:-7200}"  # max seconds to wait for holder
HOLDER_NAME="ralph-gpu"

LOG="$RALPH_DIR/loop.log"
METRICS="$RALPH_DIR/metrics.csv"
LAST_JSON="$RALPH_DIR/last_iter.json"
PROGRESS="$RALPH_DIR/PROGRESS.md"
PROMPT="$RALPH_DIR/PROMPT.md"

# Cancel only holder jobs THIS driver submitted; a pre-existing holder is
# left alone.
SUBMITTED_HOLDERS=""
cleanup() {
  for j in $SUBMITTED_HOLDERS; do
    echo ">>> cancelling holder job $j" | tee -a "$LOG"
    scancel "$j" 2>/dev/null
  done
}
trap cleanup EXIT

holder_running() {
  squeue -u "$USER" -n "$HOLDER_NAME" -h -t RUNNING -o %A 2>/dev/null | head -1
}

ensure_alloc() {
  local jid waited=0
  jid=$(holder_running)
  [ -n "$jid" ] && { echo ">>> holder $jid running" | tee -a "$LOG"; return 0; }

  jid=$(squeue -u "$USER" -n "$HOLDER_NAME" -h -t PENDING -o %A 2>/dev/null | head -1)
  if [ -z "$jid" ]; then
    jid=$(sbatch --parsable \
        --job-name="$HOLDER_NAME" \
        --partition=ampere --account=lcls:data \
        --exclusive --nodes=1 --gres=gpu:a100:1 \
        --time=12:00:00 \
        --output="$RALPH_DIR/holder_%j.log" \
        --wrap 'sleep infinity') || { echo "FATAL: sbatch failed" | tee -a "$LOG"; return 1; }
    SUBMITTED_HOLDERS="$SUBMITTED_HOLDERS $jid"
    echo ">>> submitted holder job $jid, waiting for it to start" | tee -a "$LOG"
  else
    echo ">>> holder $jid pending, waiting for it to start" | tee -a "$LOG"
  fi

  while [ -z "$(holder_running)" ]; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge "$ALLOC_WAIT_MAX" ]; then
      echo "FATAL: holder not RUNNING after ${ALLOC_WAIT_MAX}s" | tee -a "$LOG"
      return 1
    fi
  done
  echo ">>> holder $(holder_running) running (waited ${waited}s)" | tee -a "$LOG"
}

last_journal_line() {
  grep -v '^[[:space:]]*$' "$PROGRESS" | tail -1
}

record_metrics() {
  local iter="$1" wall="$2"
  python3 - "$iter" "$wall" "$LAST_JSON" "$METRICS" <<'EOF'
import csv, json, os, sys
it, wall, jpath, mpath = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    d = json.load(open(jpath))
except Exception as e:
    print(f"(metrics: could not parse {jpath}: {e})")
    sys.exit(0)
u = d.get("usage", {})
new = not os.path.exists(mpath)
with open(mpath, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["iter", "wall_s", "duration_ms", "num_turns",
                    "input_tokens", "output_tokens", "cost_usd"])
    w.writerow([it, wall, d.get("duration_ms"), d.get("num_turns"),
                u.get("input_tokens"), u.get("output_tokens"),
                d.get("total_cost_usd")])
print(f"iter {it}: wall {wall}s, turns {d.get('num_turns')}, "
      f"cost ${d.get('total_cost_usd')}")
EOF
}

echo "=== Ralph loop start $(date -u +%FT%TZ) | max=$MAX_ITERS model=$MODEL ===" | tee -a "$LOG"
malfunction=0

for ((i=1; i<=MAX_ITERS; i++)); do
  # Sentinel: the agent ends the mission with a final journal line starting
  # with LOOP DONE. Match only at line start on the last non-blank line —
  # journal prose that merely mentions the phrase ("near LOOP DONE") must not
  # end the mission.
  if last_journal_line | grep -q '^LOOP DONE'; then
    echo ">>> LOOP DONE sentinel in journal — mission complete." | tee -a "$LOG"
    break
  fi

  ensure_alloc || break

  echo "" | tee -a "$LOG"
  echo "===== iteration $i / $MAX_ITERS  $(date -u +%FT%TZ) =====" | tee -a "$LOG"
  journal_before=$(wc -l < "$PROGRESS")

  start=$(date +%s)
  claude -p --output-format json --model "$MODEL" \
      --dangerously-skip-permissions < "$PROMPT" > "$LAST_JSON" 2>>"$LOG"
  wall=$(( $(date +%s) - start ))

  # Archive this iteration's JSON before any later iteration overwrites
  # $LAST_JSON — otherwise a failed run's post-mortem trail (result text,
  # is_error, error subtype) is destroyed by the next iteration's redirect.
  cp -f "$LAST_JSON" "$RALPH_DIR/$(printf 'iter_%03d.json' "$i")" 2>/dev/null

  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('result','(no result text)'))" \
      "$LAST_JSON" >> "$LOG" 2>/dev/null || echo "(could not parse $LAST_JSON)" >> "$LOG"
  record_metrics "$i" "$wall" | tee -a "$LOG"

  journal_after=$(wc -l < "$PROGRESS")
  if [ "$journal_after" -le "$journal_before" ]; then
    malfunction=$((malfunction + 1))
    echo ">>> no journal entry appended (malfunction $malfunction/2) — see $(printf 'iter_%03d.json' "$i") for the agent's final output" | tee -a "$LOG"
    if [ "$malfunction" -ge 2 ]; then
      echo ">>> two consecutive iterations without a journal entry — stopping." | tee -a "$LOG"
      break
    fi
  else
    malfunction=0
    git --no-pager log --oneline -1 | tee -a "$LOG"
    if last_journal_line | grep -q '^BLOCKED:'; then
      # Rule 8 hand-off: if the blocker is a slurm job, wait for THAT job to
      # leave the queue in driver-side bash instead of burning a full agent
      # iteration every BLOCKED_SLEEP just to rediscover it is still pending.
      blocked_job=$(last_journal_line | grep -oE 'job [0-9]+' | head -1 | awk '{print $2}')
      if [ -n "$blocked_job" ]; then
        echo ">>> iteration blocked on job $blocked_job — polling until it leaves the queue" | tee -a "$LOG"
        bwaited=0
        while squeue -j "$blocked_job" -h -o %A 2>/dev/null | grep -q .; do
          sleep 60
          bwaited=$((bwaited + 60))
          if [ "$bwaited" -ge "$BLOCKED_WAIT_MAX" ]; then
            echo ">>> job $blocked_job still in queue after ${BLOCKED_WAIT_MAX}s — proceeding anyway" | tee -a "$LOG"
            break
          fi
        done
        if [ "$bwaited" -lt "$BLOCKED_WAIT_MAX" ]; then
          echo ">>> job $blocked_job left the queue (waited ${bwaited}s) — next iteration collects it" | tee -a "$LOG"
        fi
      else
        echo ">>> iteration blocked — sleeping ${BLOCKED_SLEEP}s before retry" | tee -a "$LOG"
        sleep "$BLOCKED_SLEEP"
      fi
    fi
  fi
done

echo "" | tee -a "$LOG"
echo "=== loop ended $(date -u +%FT%TZ) ===" | tee -a "$LOG"
[ -f "$METRICS" ] && { echo "--- metrics ---" | tee -a "$LOG"; column -s, -t < "$METRICS" | tee -a "$LOG"; }
