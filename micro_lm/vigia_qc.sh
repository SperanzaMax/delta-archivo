#!/usr/bin/env bash
# Espera a que las seis unidades de la campania de la query conjunta lleguen al paso pedido y avisa.
# El paso se lee DEL CHECKPOINT y no del JSON: la D-1 de la replica del 20-ago mostro que el JSON
# marca el paso antes de que el .pkl termine de bajar.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASO="${1:-26000}"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
PY=/home/maxi/.venv-ligamento/bin/python3

paso_de() {
  [ -f "$1" ] || { echo -1; return; }
  "$PY" -c "
import pickle,sys
try:
    d=pickle.load(open('$1','rb')); print(d.get('paso',-1))
except Exception: print(-1)
" 2>/dev/null || echo -1
}

while true; do
  listas=0; detalle=""
  for u in p3_s0 p3_s1 p3_s2 q3_s0 q3_s1 q3_s2; do
    p="$(paso_de "$AQUI/ckpts/$u.pkl")"
    [ "$p" -ge "$PASO" ] 2>/dev/null && listas=$((listas+1))
    detalle="$detalle $u=$p"
  done
  echo "$(date +%H:%M) · $listas/6 ·$detalle"
  if [ "$listas" -ge 6 ]; then
    curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
      -d chat_id="$CHAT" --data-urlencode \
      "text=✅ micro-LM · query conjunta: las 6 unidades llegaron al paso $PASO.
$detalle" >/dev/null 2>&1
    exit 0
  fi
  sleep 300
done
