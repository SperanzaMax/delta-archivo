#!/usr/bin/env bash
# Guarda una COPIA de cada checkpoint parcial que baje el rotador, para poder medir RECUP como
# trayectoria DENTRO de una unidad (`PREREG_ATRACTOR_MUDO.md`, Fase 1).
#
# El rotador pisa `ckpts/<uni>.pkl` en cada checkpoint parcial, así que la historia de los pesos se
# pierde y sólo queda el último. Los json guardan la métrica compuesta `vigente`, que en una unidad
# muda vale 0,0000 SIEMPRE y no distingue una recuperación de 0,30 de una de 0,40 — que es
# exactamente el número que hay que ver.
#
#   Uso:  archivar_traza.sh b3_s3 b3_s6 &
#
# No toca el entrenamiento ni el rotador: sólo mira el `paso` del pkl y copia cuando cambia.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/maxi/.venv-ligamento/bin/python}"
DEST="$AQUI/ckpts_traza"; mkdir -p "$DEST"
UNIS=("$@")
[ "${#UNIS[@]}" -eq 0 ] && { echo "faltan las unidades"; exit 1; }

declare -A VISTO
echo "== archivando trazas de: ${UNIS[*]} -> $DEST"

while true; do
  for U in "${UNIS[@]}"; do
    CK="$AQUI/ckpts/$U.pkl"
    [ -f "$CK" ] || continue
    # el paso sale del pkl y no del nombre: es el único lugar donde es confiable
    P="$("$PY" -c "
import pickle,sys
try:
    print(pickle.load(open('$CK','rb')).get('paso',''))
except Exception:
    print('')
" 2>/dev/null)"
    [ -z "$P" ] && continue
    [ "${VISTO[$U]:-}" = "$P" ] && continue
    # se copia sólo si el pkl no está siendo escrito: dos lecturas del tamaño con 2 s de por medio
    T1=$(stat -c %s "$CK"); sleep 2; T2=$(stat -c %s "$CK")
    [ "$T1" != "$T2" ] && continue
    cp "$CK" "$DEST/${U}_${P}.pkl" && VISTO[$U]="$P"
    echo "   [$(date +%H:%M:%S)] traza $U paso $P"
  done
  sleep 30
done
