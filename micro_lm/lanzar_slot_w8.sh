#!/usr/bin/env bash
# ENMIENDA W-8 de PREREG_SLOT_ORDEN.md (SHA de la enmienda 05e15659).
#
# El brazo que decide si el culpable del colapso fue el PESO o la INTERFAZ. Los tratamientos de las
# 18:47 corrieron con --rec-rank 1,56 y 5,45 y dejaron el logit del slot con UN valor distinto sobre
# 3072. Con `token` el mismo criterio de derivacion habia dado 0,008: entre 200 y 680 veces menos.
#
#   v03_s3 · v03_s6   --abst slot  --rec-rank 0.008   <- el peso de `token`, NO re-derivado
#
# El peso NO se re-deriva a proposito: hacerlo con el criterio que ya fallo seria el mismo error una
# segunda vez. Se toma prestado el unico que produjo un resultado legible en este proyecto.
#
# Si el logit no colapsa -> fallo la DERIVACION DEL PESO. Si colapsa igual -> fallo la INTERFAZ, y
# eso si cierra la via de la busqueda.
#
# Sembradas a mano, igual que las cuatro anteriores:
#   sembrar.py ckpts/b3_sX.pkl ckpts/v03_sX.pkl --sin-cabeza --horizonte 12000
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"

export P_NOSE=0.4 DONDE=pre BLANCO=error HORIZONTE=12000 SEMBRAR=0 \
       PERDIDA_CABEZA=recompensa REC_M=0.5 REC_F=0.2 REC_CE=1.0 REC_L=0.0 ABST=slot

LOG="$AQUI/rot_v0_0831.log"
PREFIJO=v0 REC_RANK=0.008 ACEL=t4 LOG_ROTADOR="$LOG" \
  setsid ./rotar_abst3.sh 3:3,3:6 3000 3000 250 A J F D C L K H M N I G E \
  > "$LOG" 2>&1 < /dev/null &

echo "lanzado W-8 (abst=slot, rank=0,008, unidades 3:3 y 3:6) -> $LOG"
echo "criterio W-8-a: el logit NO colapsa (mas de 10 valores distintos, saturacion < 0,95)"
