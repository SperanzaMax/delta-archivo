#!/usr/bin/env bash
# La campania de la abstencion con TRES SEMILLAS por brazo. · 2026-09-10
# Implementa ENMIENDA_ABSTENCION_20260910.md (SHA f89071b6). Doce unidades, una cuenta cada una,
# todas con el mismo instrumento (evaluacion final de 512 muestras).
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
COMUN=(--n-ent 15 --n-val 8 --n-rel 2 --batch 4 --distractores 8 --arch 8
       --p-dos 0.25 --p-una 0.35 --p-nose-rel 0.20)

#        cuenta  unidad      semilla  pasos   calentamiento
UNIDADES=(
  "C abstlargo  0 22000 3000"
  "D abstlargo  1 22000 3000"
  "E abstlargo  2 22000 3000"
  "F abstcal6p  0 19000 6000"
  "G abstcal6p  1 19000 6000"
  "H abstcal6p  2 19000 6000"
  "I abst3s     0 16000 0"
  "J abst3s     1 16000 0"
  "K abst3s     2 16000 0"
  "L abstcal3s  0 16000 3000"
  "M abstcal3s  1 16000 3000"
  "N abstcal3s  2 16000 3000"
)
for u in "${UNIDADES[@]}"; do
  read -r CUENTA UNI SEM PASOS CAL <<< "$u"
  CALARG=(); [ "$CAL" != "0" ] && CALARG=(--calentar "$CAL")
  nohup ./correr_banco_colab.sh "$CUENTA" "$UNI" "$SEM" "$PASOS" \
        "${COMUN[@]}" "${CALARG[@]}" > "logs_${UNI}_s${SEM}.log" 2>&1 &
  echo "lanzada $UNI s$SEM en la cuenta $CUENTA · $PASOS pasos · calentar $CAL · pid $!"
  sleep 4
done
