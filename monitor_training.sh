#!/bin/bash
# monitor_training.sh — background training monitor
# Usage: ./monitor_training.sh [log_file] [status_file] [interval_seconds]

LOG="${1:-/home/ubuntu/dgxsparkfinetune/output/gemma4_mini_medical/train.log}"
STATUS="${2:-/home/ubuntu/dgxsparkfinetune/output/gemma4_mini_medical/status.log}"
INTERVAL="${3:-60}"

echo "$(date '+%Y-%m-%d %H:%M:%S') [MONITOR] Starting — watching $LOG every ${INTERVAL}s" >> "$STATUS"

while true; do
    sleep "$INTERVAL"

    # Check if training process is still alive (by process name)
    FOUND=$(pgrep -f "finetune_dgx_spark" | head -1)
    if [[ -n "$FOUND" ]]; then
        ALIVE="RUNNING(pid=$FOUND)"
    else
        ALIVE="DEAD"
    fi

    # Parse tqdm training progress bar.
    # Raw tqdm format: "  7%|▋  | 41/600 [6:57:22<95:14:19, 613.34s/it]"
    # We extract each field and reformat with clear labels.
    RAW=$(grep -oP '\| *\d+/\d+ \[\d+:\d+:\d+<\d+:\d+:\d+, *[\d.]+s/it\]' "$LOG" 2>/dev/null | \
          awk -F'/' '{if(match($0,/[0-9]+\/([0-9]+)/,a) && a[1]+0>100) print}' | tail -1)

    if [[ -n "$RAW" ]]; then
        # Extract: curr/total, elapsed, ETA, s/it
        CURR=$(echo "$RAW"  | grep -oP '^\| *\K\d+(?=/)')
        TOTAL=$(echo "$RAW" | grep -oP '\d+(?= \[)')
        ELAPSED=$(echo "$RAW" | grep -oP '\[\K[\d:]+(?=<)')
        ETA_RAW=$(echo "$RAW" | grep -oP '<\K[\d:]+(?=,)')
        SPIT=$(echo "$RAW"  | grep -oP '[\d.]+(?=s/it\])')

        # Convert ETA from H:MM:SS or HH:MM:SS to "Xd Yh Zm"
        ETA_SECS=$(echo "$ETA_RAW" | awk -F: '{
            if(NF==3) print ($1*3600 + $2*60 + $3)
            else if(NF==2) print ($1*60 + $2)
            else print $1
        }')
        ETA_PRETTY=$(awk -v s="$ETA_SECS" 'BEGIN{
            d=int(s/86400); h=int((s%86400)/3600); m=int((s%3600)/60)
            if(d>0) printf "%dd %dh %dm", d, h, m
            else if(h>0) printf "%dh %dm", h, m
            else printf "%dm", m
        }')

        # Elapsed: same conversion
        EL_SECS=$(echo "$ELAPSED" | awk -F: '{
            if(NF==3) print ($1*3600 + $2*60 + $3)
            else if(NF==2) print ($1*60 + $2)
            else print $1
        }')
        EL_PRETTY=$(awk -v s="$EL_SECS" 'BEGIN{
            d=int(s/86400); h=int((s%86400)/3600); m=int((s%3600)/60)
            if(d>0) printf "%dd %dh %dm", d, h, m
            else if(h>0) printf "%dh %dm", h, m
            else printf "%dm", m
        }')

        STEP_INFO="step:${CURR}/${TOTAL} | elapsed:${EL_PRETTY} | ETA:${ETA_PRETTY} | ${SPIT}s/step"
    else
        STEP_INFO="step:?/?"
    fi

    # Read loss from TensorBoard events (Trainer logs there, not to stdout)
    TFEVENTS=$(find "$(dirname "$LOG")/runs" -name "events.out.tfevents.*" 2>/dev/null | head -1)
    if [[ -n "$TFEVENTS" ]]; then
        LATEST_LOSS=$(/home/ubuntu/venv/bin/python3 -c "
import sys
try:
    import tensorflow as tf
    best_step, best_loss, best_gnorm, best_lr = 0, None, None, None
    for e in tf.compat.v1.train.summary_iterator('$TFEVENTS'):
        for v in e.summary.value:
            if v.tag == 'train/loss':
                best_step, best_loss = e.step, v.simple_value
            if v.tag == 'train/grad_norm':
                best_gnorm = v.simple_value
            if v.tag == 'train/learning_rate':
                best_lr = v.simple_value
    if best_loss is not None:
        lr_str = f'{best_lr:.2e}' if best_lr is not None else '?'
        gn_str = f'{best_gnorm:.1f}' if best_gnorm is not None else '?'
        print(f'loss={best_loss:.4f}@step{best_step}(lr={lr_str},gnorm={gn_str})')
    else:
        print('loss=N/A')
except Exception as ex:
    print('loss=N/A')
" 2>/dev/null)
    fi
    [[ -z "$LATEST_LOSS" ]] && LATEST_LOSS="loss=N/A"

    # RAM usage
    RAM=$(free -h | awk '/^Mem:/{print $3"/"$2}')

    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    echo "${TIMESTAMP} | ${ALIVE} | RAM:${RAM} | ${LATEST_LOSS} | ${STEP_INFO:-step=?}" | tee -a "$STATUS"

    # Alert if dead
    if [[ "$ALIVE" == "DEAD" ]]; then
        echo "${TIMESTAMP} [ALERT] Training process is DEAD — check $LOG for errors" | tee -a "$STATUS"
    fi
done
