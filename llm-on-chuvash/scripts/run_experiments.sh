#!/usr/bin/env bash
# Короткие сравнительные прогоны (эксперименты для ДЗ).
# Запускать ПОСЛЕ основного обучения: каждый прогон ест все ядра.
#   bash scripts/run_experiments.sh [steps]
# Сетка: словарь 4096 / 8192 / 16384 (same seed, same шаги) + SWA на 8192.
set -euo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:-1200}"
PY=./.venv/bin/python

# 1) датасеты под альтернативные словари (8192 уже есть в data/tokenized)
for v in 4096 16384; do
  if [ ! -f "data/tokenized-v$v/train.npy" ]; then
    echo "== tokenize vocab=$v =="
    $PY -I scripts/tokenize_data.py data/corpus/cvwiki.txt "tokenizer/bpe-$v.json" "data/tokenized-v$v" 0.01
  fi
done

# 2) короткие прогоны: имя, data-dir, tokenizer, extra-args...
run() {
  local name=$1 data_dir=$2 tok=$3; shift 3
  echo "== exp $name =="
  nice -n 15 $PY -I scripts/train.py --data-dir "$data_dir" --tokenizer "$tok" \
    --out-dir checkpoints --exp-name "$name" --max-steps "$STEPS" \
    --eval-every 200 --threads "${THREADS:-8}" --dtype fp32 "$@" \
    > "checkpoints/$name.log" 2>&1
  tail -1 "checkpoints/$name.log"
}

run v4096  data/tokenized-v4096  tokenizer/bpe-4096.json
run v8192  data/tokenized       tokenizer/bpe-8192.json
run v16384 data/tokenized-v16384 tokenizer/bpe-16384.json
run swa128 data/tokenized       tokenizer/bpe-8192.json --swa-window 128

# 3) сводная таблица
echo "== summary (последний val_loss каждого прогона) =="
for name in v4096 v8192 v16384 swa128; do
  printf "%-8s %s\n" "$name" "$(tail -1 "checkpoints/$name/log.jsonl")"
done
