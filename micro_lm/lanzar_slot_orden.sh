#!/usr/bin/env bash
# PREREG_SLOT_ORDEN.md (SHA b7471e02) · el slot nulo con gradiente de ORDEN.
#
# CUATRO unidades, porque el CONTROL A ya esta corrido (`r03_s3` y `r03_s6`, interfaz `token`):
#
#   k03_s3 · k03_s6   CONTROL B    --abst slot  --rec-rank 0,0    <- separa el slot del orden
#   w03_s3            TRATAMIENTO  --abst slot  --rec-rank 1,56
#   w03_s6            TRATAMIENTO  --abst slot  --rec-rank 5,45
#
# EL PESO VA POR SEMILLA, y no es un capricho: el criterio declarado —igualar el |g| del termino de
# orden en `k_nulo` con el |g| de la perdida base en `kw`, en el ckpt de siembra— dio 1,5617 en s3 y
# 5,4459 en s6, o sea DISPERSION 3,49x, cuando el mismo criterio con `token` daba 1,12x sobre estos
# mismos dos checkpoints. Promediar daria 3,5, un numero que no describe a ninguna de las dos
# unidades. Ver §3.1 del prereg y `INFORME_COMPUERTA_SLOT_20260831.md`.
#
# Por eso son TRES rotadores y no uno: `REC_RANK` viaja como variable de entorno para toda la
# corrida, asi que dos pesos distintos no caben en el mismo rotador. Las listas de cuentas van
# DISJUNTAS —aunque el lock por cuenta ya lo cubriria— para que los tres avancen en paralelo sin
# saltearse entre si.
#
# SEMBRAR=0: las cuatro YA fueron sembradas a mano con
#     sembrar.py ckpts/b3_sX.pkl ckpts/<pre>3_sX.pkl --sin-cabeza --horizonte 12000
# porque el origen es `b3_s*` y no el `n3_s*` que el rotador buscaria solo. Con `--sin-cabeza` el
# checkpoint entra sin cabeza `abst` y `entrenar.py` la recrea fresca y reinicia Adam, que es
# exactamente lo que hizo el CONTROL A. Verificado en un smoke local de 6 pasos en CPU: arranca,
# ninguna guarda aborta, y la abstencion sale en 0,1777 —dentro de la banda de W-6, no en un extremo—.
#
# OJO con las comillas: la ruta tiene un espacio (`Nuevo Transformer`) y pasarla pelada a `env` falla
# creando igual el .log. Le costo un exit 127 al del 30-ago.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"

export P_NOSE=0.4 DONDE=pre BLANCO=error HORIZONTE=12000 SEMBRAR=0 \
       PERDIDA_CABEZA=recompensa REC_M=0.5 REC_F=0.2 REC_CE=1.0 REC_L=0.0 ABST=slot

# El nombre del checkpoint sale de `${PREFIJO}${NIVEL}_s${SEM}`, asi que los DOS tratamientos llevan
# prefijo `w0` —dan `w03_s3` y `w03_s6`, que es lo que se sembro— y lo unico que los distingue es la
# etiqueta del log. Poner `w3` y `w6` habria buscado `w33_s3` y `w63_s6`, que no existen, y el
# rotador habria arrancado de cero sin la siembra y sin decirlo.
lanzar () {   # $1 etiqueta del log · $2 prefijo · $3 unidades · $4 rec_rank · $5... cuentas
  local et="$1" pre="$2" uni="$3" rank="$4"; shift 4
  local log="$AQUI/rot_${et}_0831.log"
  PREFIJO="$pre" REC_RANK="$rank" ACEL=t4 LOG_ROTADOR="$log" \
    setsid ./rotar_abst3.sh "$uni" 3000 3000 250 "$@" > "$log" 2>&1 < /dev/null &
  echo "lanzado $et (prefijo $pre, abst=slot, rank=$rank, unidades $uni) -> $log"
}

lanzar k0 k0 3:3,3:6 0.0  A J F D C
lanzar w3 w0 3:3     1.56 L K H M
lanzar w6 w0 3:6     5.45 N I G E

echo
echo "control:  tail -f $AQUI/rot_*_0831.log"
echo "criterios (PREREG_SLOT_ORDEN.md §4):"
echo "  W-1 AUC del logit de abstencion > 0,65 en las dos semillas (token+orden dio 0,6620 y 0,6681)"
echo "  W-2 el tratamiento supera al CONTROL B por >= 0,05  <- la unica que atribuye el efecto al ORDEN"
echo "  W-3 la fraccion pegada al clip baja de 0,50 (siembra: 0,8438 en s3 y 0,6250 en s6)"
echo "  W-6 PRECONDICION: abstencion estrictamente entre 0,05 y 0,95, o W-1/W-2/W-4 son NO EVALUABLES"
