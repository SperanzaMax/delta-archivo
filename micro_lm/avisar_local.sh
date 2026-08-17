#!/usr/bin/env bash
# Vigila un tramo que corre EN LA PC y avisa a Maxi por Telegram cuando termina o muere.
#
#   Uso:  avisar_local.sh <pid> <nivel:semilla> <log>
#
# Hermano de avisar_telegram.sh, que sirve para los tramos de Colab. Aca no hay sesion que se
# pierda: el modo de falla es que el proceso muera (OOM, termica, un kill) sin haber llegado al
# tope. Por eso el disparador es el pid, no una linea del log.
set -uo pipefail

PID="${1:?falta el pid}"
UNIDAD="${2:?falta nivel:semilla}"
LOG="${3:?falta el log}"
NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
TOPE="${TOPE:-12000}"
JS="$AQUI/corridas_$(date +%Y%m%d)/n${NIVEL}_s${SEM}.json"

mandar() {
  curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null
}

while [ -d "/proc/$PID" ]; do sleep 60; done

PASO="$(grep -o '"paso": [0-9]*' "$JS" 2>/dev/null | tail -1 | grep -o '[0-9]*')"
VIG="$(grep -o '"vigente": [0-9.]*' "$JS" 2>/dev/null | tail -1 | cut -d' ' -f2 | cut -c1-6)"
ANT="$(grep -o '"anterior": [0-9.]*' "$JS" 2>/dev/null | tail -1 | cut -d' ' -f2 | cut -c1-6)"
TEMP="$(sensors 2>/dev/null | grep -m1 'Package id 0' | grep -o '+[0-9.]*°C' | head -1)"

if [ "${PASO:-0}" -ge "$TOPE" ] 2>/dev/null; then
  mandar "✅ micro-LM LOCAL · n${NIVEL}_s${SEM}: COMPLETA en el paso ${PASO} de ${TOPE}.
vigente ${VIG:-?} · anterior ${ANT:-?} · PC ${TEMP:-?}"
else
  mandar "⚠️ micro-LM LOCAL · n${NIVEL}_s${SEM}: el proceso termino ANTES del tope.
Ultimo paso: ${PASO:-0} de ${TOPE} · PC ${TEMP:-?}
El checkpoint es continuable, asi que se retoma desde ahi. Ultimas lineas:
$(tail -3 "$LOG" 2>/dev/null)"
fi
