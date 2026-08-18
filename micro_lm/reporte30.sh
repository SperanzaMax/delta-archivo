#!/usr/bin/env bash
# Manda a Telegram, cada N minutos, como viene la campania. Solo LEE ARCHIVOS LOCALES.
#
#   Uso:  reporte30.sh <unidades> <pasos> [minutos]
#   Ej.:  reporte30.sh 3:1,4:2 12000 30
#
# Por que no consulta el CLI de Colab: mientras un lanzador esta polleando una cuenta, cualquier
# otro proceso `colab` sobre esa misma cuenta se pisa el sessions.json y deja la VM inalcanzable
# (costo dos VMs el 9-ago). El estado ya viaja a la PC por streaming —los JSON se reescriben en cada
# evaluacion y el checkpoint baja cada ~8 min—, asi que mirar el disco alcanza y no arriesga nada.
set -uo pipefail

UNIDADES="${1:?faltan las unidades}"
PASOS="${2:?faltan los pasos}"
MIN="${3:-30}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

# Ultimo bloque de metricas de una unidad, buscando en los JSON de todos los dias y quedandose con
# el que llego mas lejos (una unidad puede haber avanzado ayer en Colab y hoy en la PC).
estado_de() {
  local n="${1%%:*}" s="${1##*:}" mejor=0 arch=""
  for f in "$AQUI"/corridas_*/n${n}_s${s}.json; do
    [ -f "$f" ] || continue
    local q; q="$(grep -o '"paso": [0-9]*' "$f" | tail -1 | grep -o '[0-9]*')"
    [ -n "$q" ] && [ "$q" -ge "$mejor" ] && { mejor="$q"; arch="$f"; }
  done
  if [ -z "$arch" ]; then echo "n${n}_s${s}: sin datos"; return; fi
  local vig ant
  vig="$(grep -o '"vigente": [0-9.]*' "$arch" | tail -1 | cut -d' ' -f2 | cut -c1-6)"
  ant="$(grep -o '"anterior": [0-9.]*' "$arch" | tail -1 | cut -d' ' -f2 | cut -c1-6)"
  local pct=$(( mejor * 100 / PASOS ))
  local marca="⏳"; [ "$mejor" -ge "$PASOS" ] && marca="✅"
  echo "$marca n${n}_s${s}: paso $mejor/$PASOS (${pct}%) · vig ${vig:-?} · ant ${ant:-?}"
}

while true; do
  LINEAS=""
  for u in $(echo "$UNIDADES" | tr ',' ' '); do
    LINEAS="$LINEAS
$(estado_de "$u")"
  done

  T="$("$AQUI/termica.sh" 2>/dev/null | grep -o '[0-9]*' | head -1)"
  ROT="$(pgrep -c -f "rotar[_]tramos.sh" 2>/dev/null || echo 0)"
  LOC="$(pgrep -c -f "entrenar[.]py" 2>/dev/null || echo 0)"
  # Que esta probando el rotador ahora mismo, leido de su propio log.
  ULT="$(tail -3 "$AQUI/logs_campania_$(date +%Y%m%d)/rotador.log" 2>/dev/null \
         | grep -oE '(vuelta [0-9]+ · cuenta [A-N]|OTORGADA en [A-N]|tramo [0-9]:[0-9])' | tail -1)"

  mandar "📊 micro-LM · $(date +%H:%M)
$LINEAS

CPU $T °C · rotador Colab: $([ "$ROT" -gt 0 ] && echo "vivo (${ULT:-arrancando})" || echo "PARADO")
entrenamiento local: $([ "$LOC" -gt 0 ] && echo "corriendo" || echo "parado")"

  sleep $(( MIN * 60 ))
done
