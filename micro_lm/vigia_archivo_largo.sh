#!/usr/bin/env bash
# VIGIA de la campania del archivo largo · PREREG_ARCHIVO_LARGO (c769a4ef) + ENMIENDA (0410e957)
#
#   Uso:  vigia_archivo_largo.sh [minutos_entre_avisos] [pasos_meta]
#
# Manda a Telegram la foto de las 6 unidades cada tanto, y un aviso final cuando las 6 llegan.
# Va como proceso aparte a proposito: los avisos tienen que llegar aunque nadie este mirando la
# terminal, que es justo cuando sirven.
set -uo pipefail

CADA_MIN="${1:-25}"
META="${2:-2000}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$AQUI/tg_token.sh"
PY="${PY:-/home/maxi/.venv-ligamento/bin/python}"
UNIDADES="lg3_s0 lg3_s1 lg3_s2 lc3_s0 lc3_s1 lc3_s2"

mandar() {
  curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
    -d "chat_id=${TG_CHAT}" -d "parse_mode=HTML" --data-urlencode "text=$1" >/dev/null
}

foto() {
  "$PY" - "$AQUI" "$META" $UNIDADES <<'PY'
import json, os, sys, glob
aqui, meta = sys.argv[1], int(sys.argv[2])
filas, listas = [], 0
for u in sys.argv[3:]:
    js = sorted(glob.glob(os.path.join(aqui, "corridas_*", f"{u}.json")))
    if not js:
        filas.append(f"· {u}: sin arrancar"); continue
    d = json.load(open(js[-1]))
    p = d.get("paso", 0)
    if p >= meta:
        listas += 1
    corto = d.get("cruzada_corto") or d.get("cruzada_largo") or {}
    filas.append(f"· <b>{u}</b> paso {p}/{meta} · largo {d.get('vigente', float('nan')):.4f} "
                 f"· corto {corto.get('vigente', float('nan')):.4f}")
print("\n".join(filas))
print(f"LISTAS={listas}")
PY
}

mandar "👁 Vigía del archivo largo armado · foto cada ${CADA_MIN} min · meta ${META} pasos"
while true; do
  sleep $((CADA_MIN * 60))
  F="$(foto)"
  N="$(echo "$F" | grep -o 'LISTAS=[0-9]*' | cut -d= -f2)"
  T="$(echo "$F" | grep -v LISTAS=)"
  if [ "${N:-0}" -ge 6 ]; then
    mandar "🟢 <b>Campaña del archivo largo COMPLETA</b> · las 6 unidades en ${META} pasos

${T}

Midiendo L-4 sobre las tres tratadas (masa_turnos con su control de barajado)…"
    # L-4 se mide aca y no antes: necesita los checkpoints terminados.
    for u in lg3_s0 lg3_s1 lg3_s2; do
      "$PY" "$AQUI/masa_turnos.py" "$AQUI/ckpts/$u.pkl" --ses-extra 26 --barajar --lotes 4         --salida "$AQUI/corridas_$(date +%Y%m%d)/${u}_masa.json" >> "$AQUI/salidas/vigia_al.log" 2>&1
    done
    J="$("$PY" "$AQUI/juzgar_archivo_largo.py" --meta "$META" 2>&1)"
    mandar "🧾 <b>Criterios del archivo largo</b>
<pre>${J}</pre>
Ojo: seis veces un juez automático dio un veredicto incorrecto en este proyecto. Los números están para leerse, no el booleano."
    break
  fi
  mandar "📊 <b>Archivo largo</b> · ${N}/6 en ${META}

${T}"
done
