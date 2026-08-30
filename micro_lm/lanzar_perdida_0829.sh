#!/usr/bin/env bash
# `PREREG_PERDIDA_CABEZA.md` (SHA 0f57609d) + `ENMIENDA_PERDIDA_CABEZA.md` (SHA fe058151).
#
# Dos condiciones nuevas contra el control `b3_s3`..`b3_s8`, que son las SEIS unidades sin base.
#
#   bl3_sX   --perdida-cabeza balance   (BCE pesada por el inverso de la frecuencia de clase)
#   rk3_sX   --perdida-cabeza ranking   (sustituto del AUC por pares)
#
# SEMBRAR=0 es lo que la enmienda decide y es el motivo de que este script exista. El default del
# rotador es 1, y con 1 las semillas 0..2 arrancarian desde `n3_sX.pkl` (12000 pasos, RECUP ~0,78) y
# las demas de cero, que es exactamente el confound que se encontro hoy en la campania de control.
# Aca no hay bases para 3..8 de todos modos, pero se declara igual y queda en el log: la homogeneidad
# no puede depender de que un archivo no exista.
#
# Y va como script por lo de siempre, el espacio en "Nuevo Transformer" rompe `env $COM ... $PWD/...`
# sin comillas (D-3 de DESVIACIONES_TASA_REGIMEN.md).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"

export P_NOSE=0.4 ABST=cabeza DONDE=pre BLANCO=error HORIZONTE=26000 SEMBRAR=0

# 3000 pasos, TRAMO=3000 de una sola vez. Con TRAMO=3000 el presupuesto de polling es
# MIN = 3*10+20 = 50 min y los 3000 pasos tardan ~36, asi que casi no se desperdicia VM — a
# diferencia de TRAMO=2000, que tiraba ~16 min por tramo.
PASOS=3000; TRAMO=3000; CADA=250

lanzar() {  # $1 prefijo · $2 perdida · $3 acelerador · $4 log · $5... cuentas
  local pre="$1" per="$2" acel="$3" log="$4"; shift 4
  PREFIJO="$pre" PERDIDA_CABEZA="$per" ACEL="$acel" LOG_ROTADOR="$AQUI/$log" \
    setsid ./rotar_abst3.sh 3:3,3:6 "$PASOS" "$TRAMO" "$CADA" "$@" \
    > "$AQUI/$log" 2>&1 < /dev/null &
  echo "$pre ($per, $acel) lanzado -> $log"
}

# Cuatro rotadores, dos unidades cada uno por el bug de la tercera sesion (§4.1 del ESTADO del 28),
# y listas de cuentas disjuntas para que no se peleen el lock.
PREFIJO=bl PERDIDA_CABEZA=balance ACEL=tpu LOG_ROTADOR="$AQUI/rot_bl_a_0829.log" \
  setsid ./rotar_abst3.sh 3:3,3:6 $PASOS $TRAMO $CADA A J F > "$AQUI/rot_bl_a_0829.log" 2>&1 < /dev/null &
echo "bl 3:3,3:6 (balance, tpu) lanzado"
sleep 3
PREFIJO=bl PERDIDA_CABEZA=balance ACEL=t4 LOG_ROTADOR="$AQUI/rot_bl_b_0829.log" \
  setsid ./rotar_abst3.sh 3:7,3:8 $PASOS $TRAMO $CADA L K H > "$AQUI/rot_bl_b_0829.log" 2>&1 < /dev/null &
echo "bl 3:7,3:8 (balance, t4) lanzado"
sleep 3
PREFIJO=rk PERDIDA_CABEZA=ranking ACEL=tpu LOG_ROTADOR="$AQUI/rot_rk_a_0829.log" \
  setsid ./rotar_abst3.sh 3:3,3:6 $PASOS $TRAMO $CADA D C E > "$AQUI/rot_rk_a_0829.log" 2>&1 < /dev/null &
echo "rk 3:3,3:6 (ranking, tpu) lanzado"
sleep 3
PREFIJO=rk PERDIDA_CABEZA=ranking ACEL=t4 LOG_ROTADOR="$AQUI/rot_rk_b_0829.log" \
  setsid ./rotar_abst3.sh 3:7,3:8 $PASOS $TRAMO $CADA M N I > "$AQUI/rot_rk_b_0829.log" 2>&1 < /dev/null &
echo "rk 3:7,3:8 (ranking, t4) lanzado"
sleep 3
# s4 y s5 son las dos unidades de P-0 (no-danio) y van juntas en un quinto rotador.
PREFIJO=bl PERDIDA_CABEZA=balance ACEL=t4 LOG_ROTADOR="$AQUI/rot_bl_c_0829.log" \
  setsid ./rotar_abst3.sh 3:4,3:5 $PASOS $TRAMO $CADA G > "$AQUI/rot_bl_c_0829.log" 2>&1 < /dev/null &
echo "bl 3:4,3:5 (balance, t4) lanzado"
sleep 3
PREFIJO=rk PERDIDA_CABEZA=ranking ACEL=tpu LOG_ROTADOR="$AQUI/rot_rk_c_0829.log" \
  setsid ./rotar_abst3.sh 3:4,3:5 $PASOS $TRAMO $CADA F > "$AQUI/rot_rk_c_0829.log" 2>&1 < /dev/null &
echo "rk 3:4,3:5 (ranking, tpu) lanzado"
