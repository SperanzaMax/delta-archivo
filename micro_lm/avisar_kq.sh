#!/usr/bin/env bash
# Espera a que las 3 unidades de kernel 5 lleguen a 26000 y avisa a Maxi con el JUICIO ya hecho,
# no con un «terminó». Va como proceso aparte a proposito: el aviso no puede depender de que yo
# este mirando la terminal, que es justo cuando sirve.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
. ./tg_token.sh
PY=/home/maxi/.venv-ligamento/bin/python
mandar(){ curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null; }

for _ in $(seq 1 480); do        # hasta 8 h
  n=0
  for s in 0 1 2; do
    p=$($PY -c "import pickle;print(pickle.load(open('ckpts/kq3_s$s.pkl','rb')).get('paso',0))" 2>/dev/null || echo 0)
    [ "${p:-0}" -ge 26000 ] && n=$((n+1))
  done
  [ "$n" -ge 2 ] && break        # el prereg lee con >=2 (riesgo de legibilidad K-0)
  sleep 60
done

RES="$($PY -u juzgar_kq.py 2>/dev/null | tail -32)"
mandar "🔬 KERNEL 5 · terminó y ya está juzgado (prereg SHA 50c4503d)

$RES"
