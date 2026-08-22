#!/usr/bin/env bash
# Rota entre cuentas corriendo TRAMOS que reanudan desde checkpoint. Sirve para las DOS familias.
#
#   Uso:  rotar_tramos2.sh <unidades> <pasos> <tramo> <cada> [cuentas...]
#   Ej.:  PREFIJO=x P_NOSE=0.4 rotar_tramos2.sh 4:1 14000 2000 250 H K L
#
# Que agrega sobre rotar_tramos.sh (2026-08-17, dos pedidos de Maxi):
#
#  1. FAMILIA. `rotar_tramos.sh` tenia "n${nivel}_s${semilla}" escrito a mano en las rutas, asi que
#     sólo servia para la campaña base. Aca la familia sale de PREFIJO —"n" campaña base, "x"
#     campaña de abstencion— igual que en tramo_colab.sh, y P_NOSE viaja al entrenamiento.
#
#  2. NUNCA ESPERAR A UNA CUENTA. «si alguna cuenta no le otorgan t4 cambia a otra automaticamente,
#     no te quedes esperando». Un 503 pasa a la cuenta siguiente en el acto: cero espera dentro de
#     la vuelta. Sólo se descansa al agotar la lista entera, que es cuando esperar significa algo
#     distinto —que no hay T4 en ninguna— y no simplemente que esta cuenta esta seca.
#
#  3. LOCK POR CUENTA. Dos procesos `colab` sobre la misma cuenta se pisan el sessions.json y dejan
#     la VM inalcanzable (costo dos VMs el 9-ago). Con dos rotadores vivos a la vez el riesgo deja
#     de ser teorico, asi que cada uno toma ~/.colab-pool/en_uso_<LETRA> antes de tocar la cuenta y
#     lo suelta al terminar; si el lock esta tomado por un proceso VIVO, saltea la cuenta.
set -uo pipefail

UNIDADES="${1:?faltan las unidades, p.ej. 4:1}"
PASOS="${2:?faltan los pasos totales}"
TRAMO="${3:?falta el tramo}"
CADA="${4:?falta el cada}"
shift 4
CUENTAS=("$@")
[ "${#CUENTAS[@]}" -eq 0 ] && CUENTAS=(H K L M N I G C D E F J A)

export PREFIJO="${PREFIJO:-n}"
export P_NOSE="${P_NOSE:-0.0}"
export ABST="${ABST:-token}"
export REINIT="${REINIT:-1}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
CKPTS="$AQUI/ckpts"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
POOL="$HOME/.colab-pool"; mkdir -p "$POOL"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
VUELTAS="${VUELTAS:-8}"
ESPERA_VUELTA="${ESPERA_VUELTA:-600}"
mkdir -p "$SALIDA" "$CKPTS"

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

# Red de seguridad, arriba del `timeout -k` de cada llamada. El 19-ago un tramo estuvo 3h47 colgado
# contra una VM que ya no existia y trabo la campaña entera, porque el rotador espera al tramo. La
# causa de fondo era que los `timeout` mandaban SIGTERM y nada mas —ya corregido con `-k 30`— pero
# el watchdog cubre lo que un timeout por llamada no puede: que el tramo se cuelgue SIN estar dentro
# de una llamada con timeout. Se lanza solo y muere con el rotador.
#
# El log que se vigila sale del PROPIO stdout, no de una ruta supuesta. El 20-ago el rotador se
# lanzo con el stdout redirigido a otro archivo: `$SALIDA/rotador.log` quedo en 0 bytes, el watchdog
# midio SU mtime, lo vio quieto y mato un tramo sano a los 12 min exactos —dos veces habria bastado
# para gastar la cuota del dia en 1250 pasos por cuenta—. El watchdog hacia lo que le pedimos; lo
# que estaba mal era que se le pasaba un archivo que nadie escribia. `/proc/$$/fd/1` dice a donde va
# la salida de verdad, asi que la vigilancia deja de depender de como se invoco el rotador.
#
# Va `$$` y no `self` a proposito: dentro de `$(...)` el que corre es `readlink`, y `/proc/self` es
# EL, con su stdout enganchado al pipe de la sustitucion de comandos. Preguntando por `self` la
# deteccion contesta siempre «pipe» y nunca encuentra el archivo. `$$` es el shell del rotador.
# 2026-08-21: pasa al v2. El v1 elegia la victima con `pgrep | head -1` y vigilaba un solo
# log, asi que con DOS campanias podia matar el tramo sano de la otra; el v2 identifica al
# tramo por parentesco (`pgrep -P` del rotador). Y se cae el `! pgrep -f watchdog` global:
# con el v2 cada rotador arma el SUYO, y esa guarda impedia justamente eso.
if [ -x "$AQUI/watchdog_tramo2.sh" ]; then
  SALIDA_REAL="$(readlink -f /proc/$$/fd/1 2>/dev/null || true)"
  if [ -n "${LOG_ROTADOR:-}" ]; then
    LOG_ROT="$LOG_ROTADOR"
  elif [ -f "$SALIDA_REAL" ]; then
    LOG_ROT="$SALIDA_REAL"                 # el stdout es un archivo regular: eso es lo que crece
  else
    LOG_ROT="$SALIDA/rotador.log"          # tty o pipe: no hay mtime que mirar, se usa el de siempre
  fi
  touch "$LOG_ROT" 2>/dev/null
  setsid nohup "$AQUI/watchdog_tramo2.sh" "$LOG_ROT" $$ >/dev/null 2>&1 < /dev/null &
  echo "== watchdog2 armado sobre $LOG_ROT (rotador $$)"
fi

uni_de() {
  local n="${1%%:*}"
  local s="${1##*:}"
  echo "${PREFIJO}${n}_s${s}"
}

completa() {
  local u; u="$(uni_de "$1")"
  grep -q "\"paso\": $PASOS" "$SALIDA/${u}.json" 2>/dev/null
}

bloqueada() {
  local u; u="$(uni_de "$1")"
  local lk="$CKPTS/${u}.local.lock"
  [ -f "$lk" ] || return 1
  local pid; pid="$(cat "$lk" 2>/dev/null)"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
  rm -f "$lk"; return 1
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

paso_de() {
  local u; u="$(uni_de "$1")"
  local p=""
  for f in "$AQUI"/corridas_*/"${u}".json; do
    [ -f "$f" ] || continue
    local q; q="$(grep -o '"paso": [0-9]*' "$f" 2>/dev/null | tail -1 | grep -o '[0-9]*')"
    [ -n "$q" ] && { [ -z "$p" ] || [ "$q" -gt "$p" ]; } && p="$q"
  done
  echo "${p:-0}"
}

# --- lock por cuenta ----------------------------------------------------------------------------
tomar_cuenta() {
  local lk="$POOL/en_uso_$1"
  if [ -f "$lk" ]; then
    local pid; pid="$(cat "$lk" 2>/dev/null)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then return 1; fi
    rm -f "$lk"          # lock rancio
  fi
  echo $$ > "$lk"; return 0
}
soltar_cuenta() { rm -f "$POOL/en_uso_$1"; }
trap 'for c in "${CUENTAS[@]}"; do [ "$(cat "$POOL/en_uso_$c" 2>/dev/null)" = "$$" ] && rm -f "$POOL/en_uso_$c"; done' EXIT

cli_de() {
  if [ "$1" = "A" ]; then
    CL=( "$COLAB" --auth adc ); unset CLOUDSDK_CONFIG
  else
    export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$1"
    CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$1.json" )
  fi
}

echo "== rotador2 · familia '$PREFIJO' · p_nose $P_NOSE · unidades $UNIDADES · cuentas: ${CUENTAS[*]}"
for v in $(seq 1 "$VUELTAS"); do
  for c in "${CUENTAS[@]}"; do
    FALTAN="$(pendientes)"
    if [ -z "$FALTAN" ]; then
      TODO=1
      for u in $(echo "$UNIDADES" | tr ',' ' '); do completa "$u" || TODO=0; done
      if [ "$TODO" = "1" ]; then
        echo "== todo completo"
        mandar "✅ micro-LM · familia '$PREFIJO': $UNIDADES llegaron al paso $PASOS."
        exit 0
      fi
      echo "-- lo pendiente corre en la PC; se revisa en 5 min"; sleep 300; continue
    fi
    [ "$c" = "A" ] || [ -f "$HOME/.gcloud-cuenta$c/application_default_credentials.json" ] || {
      echo "-- $c sin ADC, se saltea"; continue; }
    tomar_cuenta "$c" || { echo "-- $c la esta usando otro proceso, se saltea"; continue; }

    echo "-- vuelta $v · cuenta $c · faltan: $FALTAN · $(date +%H:%M:%S)"
    cli_de "$c"
    SESION="tr2_${c,,}_$(date +%H%M)"
    if ! timeout -k 30 420 "${CL[@]}" new -s "$SESION" --gpu T4 >/dev/null 2>&1; then
      echo "   503 en $c — SIGUIENTE CUENTA ya (sin esperar)"
      soltar_cuenta "$c"; continue
    fi
    HW="$(timeout -k 30 180 "${CL[@]}" status -s "$SESION" 2>&1 | tail -1)"
    echo "   >> OTORGADA en $c: $HW"
    mandar "🟢 micro-LM · '$PREFIJO' · la cuenta $c otorgó GPU (vuelta $v).
$HW
Arranca: $FALTAN"

    for u in $FALTAN; do
      completa "$u" && continue
      echo "   -> tramo $u en $c (desde el paso $(paso_de "$u"))"
      "$AQUI/tramo_abst.sh" "$c" "$SESION" "$u" "$PASOS" "$TRAMO" "$CADA"
      mandar "micro-LM · $(uni_de "$u") (cuenta $c): tramo cerrado en el paso $(paso_de "$u") de $PASOS."
    done

    timeout -k 30 180 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1 || true
    soltar_cuenta "$c"
    echo "   $SESION parada; pendientes ahora: $(pendientes)"
  done
  echo "-- vuelta $v recorrida entera sin T4 en ninguna; descanso $((ESPERA_VUELTA / 60)) min"
  sleep "$ESPERA_VUELTA"
done

echo "== fin · pendientes: $(pendientes)"
mandar "🔴 micro-LM · '$PREFIJO': se agotaron $VUELTAS vueltas por el pool. Pendientes: $(pendientes)"
