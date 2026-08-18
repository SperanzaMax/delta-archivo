#!/usr/bin/env bash
# Rota entre cuentas del pool corriendo TRAMOS que REANUDAN desde el checkpoint.
#
#   Uso:  rotar_tramos.sh <unidades> <pasos> <tramo> <cada> [cuentas...]
#   Ej.:  rotar_tramos.sh 3:1,4:2 12000 8000 500 H K L M N
#
# Por que existe (2026-08-17): ya habia un `rotar_cuentas.sh`, pero llama a `lanzar_micro3.sh`, que
# corre `entrenar.py` SIN `--ckpt` — o sea arranca de cero y tira el avance. Con n3_s1 en el paso
# 4500 y n4_s2 en el 4000 eso era perder 8500 pasos de computo. Este rotador usa `tramo_colab.sh`,
# que sube el checkpoint de la PC a la VM y reanuda bit a bit.
#
# Diferencia con `insistir_cuenta.sh`: aquel se queda en UNA cuenta esperando su ventana; este
# prueba la siguiente apenas una da 503. Las dos estrategias son validas segun de que dependa la
# sequia; se rota porque el pool es de asesores del proyecto y todas estan validadas.
#
# Invariante que NO se puede romper: un solo proceso `colab` por cuenta. Este script es secuencial a
# proposito — nunca toca dos cuentas a la vez — y mientras corre NO hay que consultar por afuera la
# cuenta que este usando (ni `sessions`, ni `status`), o se pisan el sessions.json.
set -uo pipefail

UNIDADES="${1:?faltan las unidades, p.ej. 3:1,4:2}"
PASOS="${2:?faltan los pasos totales}"
TRAMO="${3:?falta el tramo}"
CADA="${4:?falta el cada}"
shift 4
CUENTAS=("$@")
[ "${#CUENTAS[@]}" -eq 0 ] && CUENTAS=(H K L M N I G C D E F J A)

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
CKPTS="$AQUI/ckpts"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
VUELTAS="${VUELTAS:-6}"
ESPERA_VUELTA="${ESPERA_VUELTA:-600}"
mkdir -p "$SALIDA" "$CKPTS"

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

# Una unidad esta COMPLETA cuando su JSON de hoy registro la evaluacion en el paso final. Se mira el
# JSON y no el checkpoint porque el JSON es lo que llega por streaming aunque la VM se caiga despues.
completa() {
  local n="${1%%:*}" s="${1##*:}"
  grep -q "\"paso\": $PASOS" "$SALIDA/n${n}_s${s}.json" 2>/dev/null
}

# Un LOCK vivo significa que esa unidad la esta corriendo la PC. Si Colab la tomara al mismo tiempo,
# los dos escribirian el mismo ckpts/nX_sY.pkl y el que guardara ultimo pisaria al otro: se perderia
# el avance del mas rapido sin que nada avise. Por eso el rotador la saltea mientras el lock viva.
bloqueada() {
  # Ojo: las tres asignaciones NO pueden ir en un mismo `local`. Bash expande todos los argumentos
  # antes de ejecutar el comando, asi que ${n} se leeria antes de existir y con `set -u` aborta la
  # funcion en silencio — el sintoma era que `pendientes` devolvia vacio y el rotador creia que la
  # PC tenia todo tomado.
  local n="${1%%:*}"
  local s="${1##*:}"
  local lk="$CKPTS/n${n}_s${s}.local.lock"
  [ -f "$lk" ] || return 1
  local pid; pid="$(cat "$lk" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  rm -f "$lk"; return 1   # lock rancio: el proceso local murio
}

pendientes() {
  local out=()
  for u in $(echo "$UNIDADES" | tr ',' ' '); do
    completa "$u" && continue
    bloqueada "$u" && continue
    out+=("$u")
  done
  echo "${out[@]}"
}

# El paso alcanzado por una unidad, mirando primero el JSON de hoy y si no los de dias anteriores.
paso_de() {
  local n="${1%%:*}" s="${1##*:}" p=""
  for f in "$SALIDA/n${n}_s${s}.json" "$AQUI"/corridas_*/n${n}_s${s}.json; do
    [ -f "$f" ] || continue
    local q; q="$(grep -o '"paso": [0-9]*' "$f" 2>/dev/null | tail -1 | grep -o '[0-9]*')"
    [ -n "$q" ] && { [ -z "$p" ] || [ "$q" -gt "$p" ]; } && p="$q"
  done
  echo "${p:-0}"
}

cli_de() {
  if [ "$1" = "A" ]; then
    CL=( "$COLAB" --auth adc )
    unset CLOUDSDK_CONFIG
  else
    export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$1"
    CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$1.json" )
  fi
}

echo "== rotador de tramos · unidades $UNIDADES · pasos $PASOS · cuentas: ${CUENTAS[*]}"
for v in $(seq 1 "$VUELTAS"); do
  for c in "${CUENTAS[@]}"; do
    FALTAN="$(pendientes)"
    if [ -z "$FALTAN" ]; then
      # Vacio puede querer decir dos cosas muy distintas: terminamos, o lo que falta lo esta
      # corriendo la PC. En el segundo caso no se sale — se espera, porque si el proceso local
      # muere la unidad vuelve a quedar libre y este rotador tiene que estar ahi para tomarla.
      TODO=1
      for u in $(echo "$UNIDADES" | tr ',' ' '); do completa "$u" || TODO=0; done
      if [ "$TODO" = "1" ]; then
        echo "== todo completo"
        mandar "✅ micro-LM · campaña base COMPLETA: $UNIDADES llegaron al paso $PASOS."
        exit 0
      fi
      echo "-- lo pendiente lo esta corriendo la PC; espera 5 min y revisa de nuevo"
      sleep 300
      continue
    fi
    [ -f "$HOME/.gcloud-cuenta$c/application_default_credentials.json" ] || [ "$c" = "A" ] || {
      echo "-- cuenta $c sin ADC, se saltea"; continue; }

    echo "-- vuelta $v · cuenta $c · faltan: $FALTAN · $(date +%H:%M:%S)"
    cli_de "$c"
    SESION="tramo_${c,,}_$(date +%H%M)"
    if ! timeout 420 "${CL[@]}" new -s "$SESION" --gpu T4 >/dev/null 2>&1; then
      echo "   503 / sin T4 en $c — se prueba la siguiente cuenta"
      continue
    fi
    HW="$(timeout 180 "${CL[@]}" status -s "$SESION" 2>&1 | tail -1)"
    echo "   >> OTORGADA en $c: $HW"
    mandar "🟢 micro-LM · la cuenta $c otorgó GPU (vuelta $v).
$HW
Arranca: $FALTAN"

    # Todas las unidades pendientes van en la MISMA VM, una atras de otra: la asignacion ya esta
    # gastada, asi que aprovecharla entera sale gratis.
    for u in $FALTAN; do
      completa "$u" && continue
      echo "   -> tramo $u en $c (desde el paso $(paso_de "$u"))"
      "$AQUI/tramo_colab.sh" "$c" "$SESION" "$u" "$PASOS" "$TRAMO" "$CADA"
      n="${u%%:*}"; s="${u##*:}"
      mandar "micro-LM · n${n}_s${s} (cuenta $c): tramo cerrado en el paso $(paso_de "$u") de $PASOS."
    done

    timeout 180 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1 || true
    echo "   sesion $SESION parada; pendientes ahora: $(pendientes)"
  done
  echo "-- fin de la vuelta $v; espera $((ESPERA_VUELTA / 60)) min antes de repasar el pool"
  sleep "$ESPERA_VUELTA"
done

echo "== rotador fin · pendientes: $(pendientes)"
mandar "🔴 micro-LM · el rotador agotó $VUELTAS vueltas por el pool. Pendientes: $(pendientes)"
