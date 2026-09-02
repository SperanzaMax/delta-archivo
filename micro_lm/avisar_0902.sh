#!/usr/bin/env bash
# Espera a las DOS campañas del 2-sep y avisa a Maxi con el JUICIO ya hecho, no con un «terminó».
# Va como proceso aparte a proposito: el aviso no puede depender de que yo este mirando la terminal.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
. ./tg_token.sh
PY=/home/maxi/.venv-ligamento/bin/python
mandar(){ curl -s -m 30 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null; }
paso(){ $PY -c "import pickle;print(pickle.load(open('ckpts/$1.pkl','rb')).get('paso',0))" 2>/dev/null || echo 0; }

# D-1 (2-sep): el cruce corre con las semillas 1, 2 y 3, no 0, 1 y 2. `cf3_s0` se descarto porque su
# primer tramo salio con `formas_q=directa` antes de que FORMAS_Q se exportara en el pipeline.
listo(){ local n=0; for s in $2; do [ "$(paso "$1$s")" -ge 26000 ] && n=$((n+1)); done; [ "$n" -ge 3 ]; }

av_k7=0; av_cf=0; av_cl=0
for _ in $(seq 1 720); do          # hasta 12 h
  if [ "$av_k7" = 0 ] && listo k73_s "0 1 2"; then
    mandar "🔬 KERNEL 7 · las tres semillas cerraron, juzgado contra el kernel 5 (prereg eb5e1d50 §C)

$($PY -u juzgar_k7.py 2>/dev/null | tail -28)"
    av_k7=1
  fi
  if [ "$av_cf" = 0 ] && listo cf3_s "1 2 3"; then
    mandar "🔀 EL CRUCE · las tres semillas cerraron (prereg 410acd25). Esto es lo que decide si la ventana es una LEY o una propiedad del generador.

$($PY -u juzgar_cruce.py 2>/dev/null | tail -30)"
    av_cf=1
  fi
  if [ "$av_cl" = 0 ] && listo cl3_s "0 1 2"; then
    mandar "🎯 EL CONTROL QUE ADJUDICA cerró (prereg 2b480ce7). Dos formas donde la relación queda AFUERA en las dos. Si nose_rel sube igual, lo que arregla es la DIVERSIDAD; si se queda en 0,6-0,7, lo que arregla es VER la relación al menos a veces.

$($PY -u juzgar_diversidad.py 2>/dev/null | tail -28)"
    av_cl=1
  fi
  [ "$av_k7" = 1 ] && [ "$av_cf" = 1 ] && [ "$av_cl" = 1 ] && break
  sleep 60
done
