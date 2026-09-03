#!/usr/bin/env bash
# Cierra las dos unidades de lejos_relleno que quedaron cortadas: son las que hacen evaluable a G-3.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
S="${1:?falta la semilla}"
CUENTAS=(H C K I G F J L M A D E O N)
for v in 1 2 3 4 5 6; do
  for c in "${CUENTAS[@]}"; do
    [ -f "g${S}_lejos_relleno_s${S}.json" ] && { echo "== s$S LISTA"; exit 0; }
    echo "-- vuelta $v · cuenta $c · $(date +%H:%M)"
    CADA=100 N_EVAL=32 ./campana_distancia.sh "$c" 800 "g$S" "lejos_relleno:$S" 2>&1 | tail -8
  done
  echo "-- descanso 8 min"; sleep 480
done
