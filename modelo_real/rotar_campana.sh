#!/usr/bin/env bash
# Rota cuentas hasta conseguir T4 para UNA semilla completa (las 4 condiciones).
#   Uso:  rotar_campana.sh <semilla> <pasos> [cuentas...]
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
S="${1:?falta la semilla}"; PASOS="${2:-800}"
shift 2; CUENTAS=("$@"); [ "${#CUENTAS[@]}" -eq 0 ] && CUENTAS=(H C K I G F J L M A D E)
completo(){ ls "g${S}_cerca_s${S}.json" "g${S}_lejos_s${S}.json" \
               "g${S}_lejos_dos_s${S}.json" "g${S}_lejos_relleno_s${S}.json" >/dev/null 2>&1; }
for v in $(seq 1 8); do
  for c in "${CUENTAS[@]}"; do
    completo && { echo "== semilla $S COMPLETA"; exit 0; }
    echo "-- vuelta $v · cuenta $c · $(date +%H:%M)"
    CADA=100 N_EVAL=32 ./campana_distancia.sh "$c" "$PASOS" "g$S" \
        "cerca:$S" "lejos:$S" "lejos_dos:$S" "lejos_relleno:$S" 2>&1 | tail -12
  done
  completo && { echo "== semilla $S COMPLETA"; exit 0; }
  echo "-- lista agotada, descanso 8 min"; sleep 480
done
