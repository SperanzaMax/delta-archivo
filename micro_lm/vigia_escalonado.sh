#!/usr/bin/env bash
# Vigila las NUEVE unidades vivas del 23-ago y avisa por Telegram: las 3 de `lat` (w3_*, a 26000) y
# las 6 del escalonado (ed3_* dinamica y ef3_* fija, a 20000).
#
# El paso se lee DEL CHECKPOINT y no del JSON, por la D-1 de la replica del 20-ago: el JSON marca el
# paso antes de que el .pkl termine de bajar.
#
# Avisa DOS cosas distintas, porque son dos noticias distintas:
#   · cada unidad que llega a su meta, una sola vez (archivo de marcas, no se repite);
#   · un latido cada media hora con la foto entera, para no tener que preguntar.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
PY=/home/maxi/.venv-ligamento/bin/python3
MARCAS="$AQUI/.avisadas_escalonado"
LATIDO="${LATIDO:-1800}"
touch "$MARCAS"

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

paso_de() {
  [ -f "$1" ] || { echo -1; return; }
  "$PY" -c "
import pickle
try:
    d=pickle.load(open('$1','rb')); print(d.get('paso',-1))
except Exception: print(-1)
" 2>/dev/null || echo -1
}

ultimo_latido=0
while true; do
  listas=0; total=0; detalle=""
  # 2026-08-25: la lista vuelve al ESCALONADO, que es lo unico vivo. `v3`/`y3` llegaron a 26000 y ya
  # estan analizadas (INFORME_LAT2 / INFORME_SLOT), asi que vigilarlas seria ruido. Las seis del
  # escalonado se estan completando a 20000 en T4 con los dos rotadores del 25-ago; `ef3_s0` ya
  # estaba en 20000 desde el 23 y el rotador la saltea. Meta unica 20000 para las seis: comparar
  # dinamica contra fija a distinto presupuesto es el error que ya costo caro tres veces.
  for par in ed3_s0:20000 ed3_s1:20000 ed3_s2:20000 \
             ef3_s0:20000 ef3_s1:20000 ef3_s2:20000; do
    u="${par%%:*}"; meta="${par##*:}"
    p="$(paso_de "$AQUI/ckpts/$u.pkl")"
    total=$((total+1))
    detalle="$detalle
  $u = $p / $meta"
    if [ "$p" -ge "$meta" ] 2>/dev/null; then
      listas=$((listas+1))
      grep -qx "$u" "$MARCAS" || { echo "$u" >> "$MARCAS"; mandar "OK · $u llego a $meta pasos."; }
    fi
  done
  echo "$(date +%H:%M) · $listas/$total ·$(echo "$detalle" | tr '\n' ' ')"

  ahora=$(date +%s)
  if [ $((ahora - ultimo_latido)) -ge "$LATIDO" ]; then
    mandar "micro-LM · $listas de $total unidades terminadas$detalle"
    ultimo_latido=$ahora
  fi

  if [ "$listas" -ge "$total" ]; then
    mandar "TERMINO TODO: las $total unidades llegaron a su meta. Faltan los controles fijo_promedio (ep3_*), que salen del promedio que la dinamica termino usando."
    exit 0
  fi
  sleep 300
done
