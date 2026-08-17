#!/usr/bin/env bash
# Espera a que termine un tramo y le manda el resultado a Maxi por Telegram.
#
#   Uso:  avisar_telegram.sh <log del tramo> <nivel:semilla>
#
# Va como proceso aparte a proposito: si el aviso dependiera de que yo esté procesando la sesion,
# no llegaria cuando Maxi no esta mirando la terminal, que es justo cuando sirve.
# Canal verificado hoy: @Albertagente_bot (los bots Nexus y los otros dan 403, ver la memoria
# `telegram-notify`).
set -uo pipefail

LOG="${1:?falta el log}"
UNIDAD="${2:?falta nivel:semilla}"
NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
CK="$AQUI/ckpts/n${NIVEL}_s${SEM}.pkl"
JS="$AQUI/corridas_$(date +%Y%m%d)/n${NIVEL}_s${SEM}.json"

mandar() {
  curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null
}

# hasta 3 h de espera; el tramo de 4000 pasos tarda ~31 min a 0,46 s/paso
for _ in $(seq 1 360); do
  grep -qE "fin del tramo|sesion perdida" "$LOG" 2>/dev/null && break
  sleep 30
done

PASO="$(grep -o '"paso": [0-9]*' "$JS" 2>/dev/null | tail -1 | grep -o '[0-9]*')"
VIG="$(grep -o '"vigente": [0-9.]*' "$JS" 2>/dev/null | tail -1 | cut -d' ' -f2 | cut -c1-6)"
ANT="$(grep -o '"anterior": [0-9.]*' "$JS" 2>/dev/null | tail -1 | cut -d' ' -f2 | cut -c1-6)"

if grep -q "sesion perdida" "$LOG" 2>/dev/null; then
  mandar "⚠️ micro-LM · n${NIVEL}_s${SEM}: se cayo la VM a mitad del tramo.
Lo rescatado llega hasta el paso ${PASO:-0}. El checkpoint de la PC conserva el ultimo tramo bajado, asi que se continua desde ahi en otra cuenta."
elif [ -f "$CK" ]; then
  mandar "✅ micro-LM · n${NIVEL}_s${SEM}: tramo cerrado en el paso ${PASO:-?} de ${TOPE:-12000}.
vigente ${VIG:-?} · anterior ${ANT:-?}
Checkpoint bajado a la PC ($(du -h "$CK" | cut -f1)): el proximo tramo puede correr en cualquier otra cuenta."
else
  mandar "⚠️ micro-LM · n${NIVEL}_s${SEM}: el tramo termino pero NO se pudo bajar el checkpoint.
Ultimo paso registrado: ${PASO:-?}. La sesion sigue viva, se puede reintentar la bajada."
fi
