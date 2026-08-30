#!/usr/bin/env bash
# ETAPA 1 de `PREREG_RECOMPENSA.md` (SHA f1f7bb66). Compuerta W-0 abierta con 21 chequeos.
#
#   tk3_sX   --abst token   --perdida-cabeza recompensa   PRINCIPAL, sin cabeza, es la que escala
#   hd3_sX   --abst cabeza  --perdida-cabeza recompensa   contraste
#
# Semillas 3..8, las seis SIN base (ENMIENDA_PERDIDA_CABEZA). SEMBRAR=0 declarado.
# 3000 pasos, horizonte 26000 para que la curva de lr en el tramo sea la misma que la del control.
#
# La Etapa 1 decide SOLO W-1 (¿sale del silencio?). W-2, que es el criterio principal y mira la
# exactitud global contra su piso, necesita la Etapa 2 a 12000 pasos y se lanza aparte: el §4 del
# pre-registro parte el presupuesto justamente porque hoy se midio que P-4 no era decidible a 3000.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"

export P_NOSE=0.4 DONDE=pre BLANCO=error HORIZONTE=26000 SEMBRAR=0 PERDIDA_CABEZA=recompensa
PASOS=3000; TRAMO=3000; CADA=250

# $1 prefijo · $2 abst · $3 unidades · $4 acel · $5 log · $6.. cuentas
lanzar() {
  local pre="$1" abst="$2" uni="$3" acel="$4" log="$5"; shift 5
  PREFIJO="$pre" ABST="$abst" ACEL="$acel" LOG_ROTADOR="$AQUI/$log" \
    setsid ./rotar_abst3.sh "$uni" "$PASOS" "$TRAMO" "$CADA" "$@" \
    > "$AQUI/$log" 2>&1 < /dev/null &
  echo "$pre $uni ($abst, $acel) -> $log"
  sleep 3
}

lanzar tk token  3:3,3:6 tpu rot_tk_a_0829.log A J F
lanzar tk token  3:7,3:8 t4  rot_tk_b_0829.log L K H
lanzar tk token  3:4,3:5 tpu rot_tk_c_0829.log D C
lanzar hd cabeza 3:3,3:6 t4  rot_hd_a_0829.log M N I
lanzar hd cabeza 3:7,3:8 tpu rot_hd_b_0829.log E G
lanzar hd cabeza 3:4,3:5 t4  rot_hd_c_0829.log F J
