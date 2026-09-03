#!/usr/bin/env bash
# Rota cuentas hasta conseguir T4 para el smoke del scan paralelo.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
CUENTAS=(H C O K I G D E F J L M N A)
for vuelta in 1 2 3; do
  for c in "${CUENTAS[@]}"; do
    ls smoke_pscan_*.log >/dev/null 2>&1 && grep -lq "VEL PSCAN" smoke_pscan_*.log 2>/dev/null && { echo "== ya hay resultado"; exit 0; }
    echo "-- vuelta $vuelta · cuenta $c · $(date +%H:%M)"
    ./smoke_pscan_colab.sh "$c" 2>&1 | tail -20
    grep -lq "VEL PSCAN" smoke_pscan_*.log 2>/dev/null && { echo "== SMOKE LISTO en $c"; exit 0; }
  done
  echo "-- lista agotada, descanso 8 min"; sleep 480
done
