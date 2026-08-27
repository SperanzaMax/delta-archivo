#!/usr/bin/env bash
# Latido: manda a Maxi por Telegram el estado de Colab + PC cada N minutos.
#
#   Uso:  ./latido.sh [minutos]     (default 15)
#
# Va como proceso aparte para que el aviso NO dependa de que yo este procesando la sesion: sirve
# justamente cuando Maxi no esta mirando la terminal. Canal verificado: @Albertagente_bot.
set -uo pipefail

CADA=$(( ${1:-15} * 60 ))
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo
PY=/home/maxi/.venv-ligamento/bin/python
GASTO="$AQUI/gasto"; HOY="$(date +%Y%m%d)"
LOG="$AQUI/logs_campania_$(date +%Y%m%d)/base_n3_s0.log"

mandar() {
  curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null
}

paso_de() {
  [ -f "$AQUI/ckpts/$1.pkl" ] || { echo 0; return; }
  "$PY" -c "import pickle;print(pickle.load(open('$AQUI/ckpts/$1.pkl','rb'))['paso'])" 2>/dev/null || echo 0
}

while true; do
  sleep "$CADA"

  VIVAS=0; APAGADAS=""
  for c in C D E F G H I J K L M N A; do
    if [ -f "$GASTO/${HOY}_$c.off" ]; then APAGADAS="$APAGADAS$c "; else VIVAS=$(( VIVAS + 1 )); fi
  done
  DIA=0
  for f in "$GASTO/${HOY}_"[A-Z]; do [ -f "$f" ] && DIA=$(( DIA + $(cat "$f") )); done

  W=$(pgrep -c -f "worker_cola[.]sh" 2>/dev/null || echo 0)
  ULT="$(tail -3 "$LOG" 2>/dev/null | tr '\n' ' | ')"
  ACC="$(tail -3 "$GASTO/${HOY}.acc" 2>/dev/null | tr '\n' ' | ')"
  TEMP="$(sensors 2>/dev/null | grep -m1 'Package id' | grep -o '+[0-9.]*°C' | head -1)"
  CPU="$(pgrep -f 'sonda_vecino|score_archivo|control_score' >/dev/null && echo 'corriendo' || echo 'libre')"

  mandar "⏱ micro-LM $(date +%H:%M)

COLAB · worker $([ "$W" -gt 0 ] && echo VIVO || echo PARADO) · n3_s0 en $(paso_de n3_s0)/12000
cuentas vivas $VIVAS/13 · apagadas: ${APAGADAS:-ninguna}
asignaciones hoy: $DIA
aceleradores: ${ACC:-ninguno todavia}
ultimo: $ULT

PC · ${TEMP:-?} · analisis $CPU"
done
