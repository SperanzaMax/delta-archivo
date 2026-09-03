#!/usr/bin/env bash
# Rota entre cuentas hasta conseguir T4 para UNA unidad del experimento en modelo real.
# Mismo principio que el rotador del micro-LM: un «Service Unavailable» pasa a la cuenta siguiente
# en el acto, y solo se descansa al agotar la lista entera, que es cuando esperar significa algo.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
COND="${1:?falta la condicion}"
SEM="${2:?falta la semilla}"
PASOS="${3:-1200}"
shift 3
CUENTAS=("$@"); [ "${#CUENTAS[@]}" -eq 0 ] && CUENTAS=(H C O K I G D E F J L M N A)
UNI="real_${COND}_s${SEM}"

for vuelta in $(seq 1 6); do
  for c in "${CUENTAS[@]}"; do
    [ -f "$AQUI/${UNI}.json" ] && { echo "== $UNI ya esta listo"; exit 0; }
    echo "-- vuelta $vuelta · cuenta $c · $(date +%H:%M)"
    NH="${NH:-16}" ./correr_real_colab.sh "$c" "$COND" "$SEM" "$PASOS" 2>&1 | tail -25
    [ -f "$AQUI/${UNI}.json" ] && { echo "== $UNI LISTO en $c"; exit 0; }
  done
  echo "-- lista agotada, se descansa 10 min"
  sleep 600
done
echo "== $UNI NO se pudo correr"
