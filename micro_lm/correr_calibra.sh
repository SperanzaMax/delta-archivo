#!/usr/bin/env bash
# A3 (calibrar) y A4 (ensamble). EXPLORATORIO, sin prereg — ver el encabezado del script.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python
for u in p3_s0 p3_s1 p3_s2; do
  echo "=== dump $u  $(date -u +%H:%M:%S) ==="
  OMP_NUM_THREADS=2 taskset -c 0-1 "$PY" "$AQUI/sonda_calibra_ensamble.py" \
      --dump "$AQUI/ckpts/$u.pkl" --n 6000 2>&1 | grep -viE "warning"
done
echo "=== analisis $(date -u +%H:%M:%S) ==="
OMP_NUM_THREADS=2 taskset -c 0-1 "$PY" "$AQUI/sonda_calibra_ensamble.py" \
    --analizar "$AQUI/dump/p3_s0.npz" "$AQUI/dump/p3_s1.npz" "$AQUI/dump/p3_s2.npz" 2>&1 \
  | grep -viE "warning"
echo "=== fin $(date -u +%H:%M:%S) ==="
