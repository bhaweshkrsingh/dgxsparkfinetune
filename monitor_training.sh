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

    # Extract training step progress — look for x/1566 pattern (total steps)
    # tqdm line looks like: "  0%|  | 2/1566 [17:51<233:21:06, 537.13s/it]"
    TOTAL_STEPS=$(grep -oP '\d+/\d+' "$LOG" 2>/dev/null | awk -F/ '$2>100{print}' | tail -1)
    STEP_INFO=$(grep -oP '\d+%\|[^|]*\|\s*\d+/\d+ \[[\d:<, its/]+\]' "$LOG" 2>/dev/null | \
                awk -F'/' '$0~/[0-9]+\/[0-9]+/ && (match($0,/[0-9]+\/([0-9]+)/,a) && a[1]+0>100)' | tail -1)

    # Fallback: grab last line matching the training progress bar
    if [[ -z "$STEP_INFO" ]]; then
        STEP_INFO=$(grep -oP '\| *\d+/\d+ \[[\d:]+<[\d:]+, *[\d.]+s/it\]' "$LOG" 2>/dev/null | \
                    awk -F'/' '{n=match($0,/[0-9]+\/([0-9]+)/,a); if(a[1]+0>100) print}' | tail -1)
    fi

    # Latest reported loss from trainer logs
    LATEST_LOSS=$(grep -oP "'loss':\s*[\d.]+" "$LOG" 2>/dev/null | tail -1)
    if [[ -z "$LATEST_LOSS" ]]; then
        LATEST_LOSS=$(grep -oP '"loss":\s*[\d.]+"' "$LOG" 2>/dev/null | tail -1)
    fi
    if [[ -z "$LATEST_LOSS" ]]; then
        LATEST_LOSS=$(grep -oP "loss=[\d.]+" "$LOG" 2>/dev/null | tail -1)
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
