#!/usr/bin/env bash
# Corrida de `PREREG_DOS_DETECTORES.md` (SHA 91494aa0), con la desviacion D-D1 aplicada.
# UN PROCESO POR UNIDAD: `jax.jit` hornea `donde` en el trace compilado.
# taskset a dos hilos por la regla termica: XLA ignora OMP_NUM_THREADS.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python
N=${1:-6000}
for u in p3_s1 p3_s2 p3_s0 v3_s0 v3_s1 v3_s2; do
  echo "=== $u  $(date -u +%H:%M:%S) ==="
  OMP_NUM_THREADS=2 taskset -c 0-1 "$PY" "$AQUI/sonda_dos_detectores.py" \
      "$AQUI/ckpts/$u.pkl" --n "$N" --salida "$AQUI/dos_detectores/$u.json" 2>&1 \
    | grep -viE "warning|warn:"
done
echo "=== fin $(date -u +%H:%M:%S) ==="
