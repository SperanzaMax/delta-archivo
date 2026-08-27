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
export SEMBRAR="${SEMBRAR:-1}"
export CORTES="${CORTES:-}"
export REINIT="${REINIT:-1}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
CKPTS="$AQUI/ckpts"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
POOL="$HOME/.colab-pool"; mkdir -p "$POOL"
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo
VUELTAS="${VUELTAS:-8}"
ESPERA_VUELTA="${ESPERA_VUELTA:-600}"
mkdir -p "$SALIDA" "$CKPTS"

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

# Una unidad se escribe "nivel:semilla" y toma el PREFIJO global, o "prefijo/nivel:semilla" para
# traerse el suyo. Lo segundo es lo que necesitan las 18 fases del PREREG_FRONTERA: comparten nivel
# y semilla pero difieren en margen de entrada (85/90/95) y en condición (token/cabeza), así que
# conviven seis familias en la misma corrida del rotador y una sola variable global no alcanza.
uni_de() {
  local a="$1" pre="$PREFIJO"
  case "$a" in */*) pre="${a%%/*}"; a="${a#*/}";; esac
  local n="${a%%:*}"
  local s="${a##*:}"
  echo "${pre}${n}_s${s}"
}

# El paso final tampoco puede ser global en las fases: cada corte cayó en un paso distinto y el
# presupuesto de 2000 pasos (§7) es lo que se mantiene igual. `fases.tsv` manda cuando la unidad
# está ahí; si no, vale el PASOS de la línea de comandos.
pasos_de() {
  local u; u="$(uni_de "$1")"
  local p=""
  [ -f "$AQUI/fases.tsv" ] && p="$(awk -F'\t' -v u="$u" '$1==u {print $4; exit}' "$AQUI/fases.tsv")"
  echo "${p:-$PASOS}"
}

completa() {
  local u; u="$(uni_de "$1")"
  grep -q "\"paso\": $(pasos_de "$1")" "$SALIDA/${u}.json" 2>/dev/null
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

echo "== rotador-frontera · familia '$PREFIJO' · p_nose $P_NOSE · unidades $UNIDADES · cuentas: ${CUENTAS[*]}"
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
      P_U="$(pasos_de "$u")"
      PRE_U="$PREFIJO"; NS_U="$u"
      case "$u" in */*) PRE_U="${u%%/*}"; NS_U="${u#*/}";; esac
      echo "   -> tramo $u en $c (desde el paso $(paso_de "$u") de $P_U)"
      PREFIJO="$PRE_U" "$AQUI/tramo_frontera.sh" "$c" "$SESION" "$NS_U" "$P_U" "$TRAMO" "$CADA"
      mandar "micro-LM · $(uni_de "$u") (cuenta $c): tramo cerrado en el paso $(paso_de "$u") de $P_U."
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
