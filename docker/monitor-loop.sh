#!/bin/sh
# The scheduled monitoring job, as a container: run, sleep, repeat.
#
# Historical data has no meaningful "now", so the job's clock is MONITOR_AS_OF
# (e.g. 2019-11-01) when set, else today. MONITOR_INTERVAL_SECONDS defaults to a
# day; a cron-grade scheduler would add nothing a loop does not (decision 9).
set -eu
interval="${MONITOR_INTERVAL_SECONDS:-86400}"
while true; do
  as_of="${MONITOR_AS_OF:-$(date -u +%Y-%m-%d)}"
  echo "[monitor] run as of ${as_of}"
  if python -m fraud.monitoring.run --as-of "${as_of}" --write; then
    echo "[monitor] ok; next run in ${interval}s"
  else
    echo "[monitor] FAILED; retrying in ${interval}s" >&2
  fi
  sleep "${interval}"
done
