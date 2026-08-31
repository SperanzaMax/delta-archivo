#!/usr/bin/env bash
# PREREG_ORDEN_NOSE.md (SHA 9e5659e5) · romper la degeneracion con un termino de ORDEN.
#
# DOS unidades nada mas, porque el CONTROL YA ESTA CORRIDO: `t03_s3` y `t03_s6` son exactamente esta
# misma configuracion con --rec-rank 0, asi que el contraste es pareado y sale por la mitad de GPU.
#
#   r03_s3 · r03_s6   sembradas desde b3_s3 / b3_s6, --rec-rank 0,008
#
# El peso NO se eligio: sale de igualar el gradiente en la columna de NOSE con el gradiente medio del
# resto del vocabulario, medido EN EL CHECKPOINT DE SIEMBRA (6,774e-06 contra 8,418e-04 -> 0,00805 en
# s3 y 0,00720 en s6). Medirlo en el ckpt de siembra y no a mitad de corrida es la correccion al
# error del 30-ago, donde el 3,5 resulto no ser una constante.
#
# SEMBRAR=0: las dos unidades YA fueron sembradas a mano con `sembrar.py --sin-cabeza`, porque el
# origen es `b3_s*` y no el `n3_s*` que el rotador buscaria solo.
#
# OJO con las comillas, que al del 30-ago le costo un exit 127 sin lanzar nada: la ruta tiene un
# espacio (`Nuevo Transformer`) y pasarla pelada a `env` falla creando igual el .log.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"

export P_NOSE=0.4 DONDE=pre BLANCO=error HORIZONTE=12000 SEMBRAR=0 \
       PERDIDA_CABEZA=recompensa REC_M=0.5 REC_F=0.2 REC_CE=1.0

LOG="$AQUI/rot_r0_0831.log"
PREFIJO=r0 ABST=token REC_L=0.0 REC_RANK=0.008 ACEL=t4 LOG_ROTADOR="$LOG" \
  setsid ./rotar_abst3.sh 3:3,3:6 3000 3000 250 A J F D C L K H M N I G E \
  > "$LOG" 2>&1 < /dev/null &

echo "lanzado r0 (abst=token, L=0,0, RANK=0,008) -> $LOG"
echo
echo "control:  tail -f $LOG"
echo "criterio: O-1 acuerdo con «no hay respuesta» > 0,60 (control 0,4985)"
echo "          O-2 pureza por relacion < 0,70 (control 0,977-0,982)"
echo "          O-3 invento <= 0,10  <- el que puede CERRAR la linea"
