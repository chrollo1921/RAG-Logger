#!/usr/bin/env bash
# Demo: multi-file ingest on the real console log (plus a gzip "rotated" copy) into a throwaway index.
cd "$(dirname "$0")/.." || exit 1
LOG=${1:-/mnt/c/Users/boyapatikarthik/Downloads/console-20260829.log}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/node1" "$WORK/node2"
cp "$LOG" "$WORK/node1/console.log"
gzip -c "$LOG" > "$WORK/node2/console.log.1.gz"
RL="python3 -m raglogger --env-file off --engine native --index $WORK/idx.json.gz"

echo "== 1) rebuild from a directory (2 files, recursive)"
$RL ingest "$WORK"
echo; echo "== 2) sources"
$RL sources
echo; echo "== 3) append again: unchanged files are skipped"
$RL ingest --append "$WORK"
echo; echo "== 4) search across all sources vs only node2 (--source matches label or path)"
$RL search "IllegalArgumentException removing item" -k 4
$RL search "IllegalArgumentException removing item" -k 4 --source node2
echo; echo "== 5) remove one source (by label shown in 'sources')"
$RL ingest --remove console.log
$RL sources
