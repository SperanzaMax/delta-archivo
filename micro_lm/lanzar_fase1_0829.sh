#!/usr/bin/env bash
# Fase 1 de `PREREG_ATRACTOR_MUDO.md` (SHA 2be4a610). Termina `b3_s3` y `b3_s6` a 26000 con los
# flags IDENTICOS de `PREREG_TASA_REGIMEN`, o sea sin condicion nueva.
#
# Va como script y no como una linea de shell a proposito: el comando del §6 del ESTADO del 28 usa
# `env $COM ... LOG_ROTADOR=$PWD/...` sin comillas, y esta ruta tiene un ESPACIO ("Nuevo
# Transformer"), asi que `env` recibe la ruta partida en dos y muere con exit 127 antes de lanzar
# nada. Cazado al lanzar, 29-ago.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"

export PREFIJO=b P_NOSE=0.4 ABST=cabeza DONDE=pre BLANCO=error HORIZONTE=26000

ACEL=tpu LOG_ROTADOR="$AQUI/rot_s3_0829.log" \
  setsid ./rotar_abst3.sh 3:3 26000 2000 250 A J F D C \
  > "$AQUI/rot_s3_0829.log" 2>&1 < /dev/null &
echo "s3 (tpu) lanzado"

sleep 3

ACEL=t4 LOG_ROTADOR="$AQUI/rot_s6_0829.log" \
  setsid ./rotar_abst3.sh 3:6 26000 2000 250 L K H M I \
  > "$AQUI/rot_s6_0829.log" 2>&1 < /dev/null &
echo "s6 (t4) lanzado"
