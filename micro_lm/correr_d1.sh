#!/usr/bin/env bash
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python
for u in p3_s1 p3_s2 p3_s0; do
  echo "=== D1 $u  $(date -u +%H:%M:%S) ==="
  OMP_NUM_THREADS=2 taskset -c 2-3 "$PY" "$AQUI/sonda_d1.py" "$AQUI/ckpts/$u.pkl" \
      --n 6000 --salida "$AQUI/dos_detectores/d1_$u.json" 2>&1 | grep -viE "warning"
done
echo "=== fin D1 $(date -u +%H:%M:%S) ==="
