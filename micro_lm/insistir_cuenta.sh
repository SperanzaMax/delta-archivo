#!/usr/bin/env bash
# Insiste con UNA sola cuenta hasta que Colab le otorgue la GPU, y ahi mismo corre el tramo.
#
#   Uso:  insistir_cuenta.sh <CUENTA> <sesion> <nivel:semilla> <pasos> <tramo> <cada> [intentos] [espera]
#   Ej.:  insistir_cuenta.sh A base17c 4:2 12000 8000 500 24 300
#
# Por que existe (2026-08-17): la alternativa que veniamos usando —rotar por el pool hasta que
# alguna cuenta afloje— es justo lo que Colab prohibe, asi que este script se queda en la cuenta
# propia y espera. Un intento que da 503 NO gasta asignacion (medido el 14-ago), asi que insistir
# es barato; lo unico que cuesta es tiempo.
#
# Un solo proceso `colab` por cuenta: mientras esto corre, NO consultar la misma cuenta por afuera
# (ni `sessions`, ni `status`) o se pisan el sessions.json y la VM queda inalcanzable.
set -uo pipefail

CUENTA="${1:?falta la cuenta}"
SESION="${2:?falta la sesion}"
UNIDAD="${3:?falta nivel:semilla}"
PASOS="${4:?faltan los pasos}"
TRAMO="${5:?falta el tramo}"
CADA="${6:?falta el cada}"
INTENTOS="${7:-24}"
ESPERA="${8:-300}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo

mandar() {
  curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null
}

if [ "$CUENTA" = "A" ]; then
  CL=( "$COLAB" --auth adc )
else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
  CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
fi

for i in $(seq 1 "$INTENTOS"); do
  echo "== intento $i/$INTENTOS · $(date +%H:%M:%S)"
  if timeout 420 "${CL[@]}" new -s "$SESION" --gpu T4 >/dev/null 2>&1; then
    HW="$(timeout 180 "${CL[@]}" status -s "$SESION" 2>&1 | tail -1)"
    echo "== OTORGADA: $HW"
    mandar "🟢 micro-LM · cuenta $CUENTA otorgo GPU al intento $i.
$HW
Arranca el tramo ${UNIDAD} (+${TRAMO} de ${PASOS} pasos)."
    "$AQUI/tramo_colab.sh" "$CUENTA" "$SESION" "$UNIDAD" "$PASOS" "$TRAMO" "$CADA"
    NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"
    JS="$AQUI/corridas_$(date +%Y%m%d)/n${NIVEL}_s${SEM}.json"
    PASO="$(grep -o '"paso": [0-9]*' "$JS" 2>/dev/null | tail -1 | grep -o '[0-9]*')"
    VIG="$(grep -o '"vigente": [0-9.]*' "$JS" 2>/dev/null | tail -1 | cut -d' ' -f2 | cut -c1-6)"
    ANT="$(grep -o '"anterior": [0-9.]*' "$JS" 2>/dev/null | tail -1 | cut -d' ' -f2 | cut -c1-6)"
    mandar "micro-LM · n${NIVEL}_s${SEM} (cuenta $CUENTA): tramo cerrado en el paso ${PASO:-?} de ${PASOS}.
vigente ${VIG:-?} · anterior ${ANT:-?}
La sesion '$SESION' queda VIVA para encadenar el tramo siguiente sin gastar otra asignacion."
    exit 0
  fi
  echo "   503 / sin GPU — no gasta asignacion; se reintenta en ${ESPERA}s"
  sleep "$ESPERA"
done

echo "== agotados los $INTENTOS intentos sin GPU"
mandar "🔴 micro-LM · cuenta $CUENTA: $INTENTOS intentos sin que Colab otorgue T4 ($(( INTENTOS * ESPERA / 60 )) min).
No hay rotacion a otras cuentas. Opciones: esperar a que se libere la ventana, o alquilar GPU por hora."
