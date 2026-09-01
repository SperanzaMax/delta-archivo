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
# 2026-08-22 · campania de la query conjunta. Los dos viajan hasta `entrenar.py` por el tramo.
export DONDE="${DONDE:-pre}"
export SEMBRAR="${SEMBRAR:-1}"
# 2026-08-23 · escalonado (DISENO_ESCALONADO.md). Viajan hasta `entrenar.py` por el tramo, igual que
# DONDE. Sin exportarlos, el tramo usaria sus defaults y la campania dinamica correria fija.
export MEZCLA="${MEZCLA:-fija}"
export BLANCO="${BLANCO:-ausencia}"
export KERNEL_Q="${KERNEL_Q:-3}"   # 2026-09-01, INFORME_QUERY_CIEGA   # 2026-08-26, A5. Viaja hasta entrenar.py por el tramo.
# 2026-08-29, PREREG_PERDIDA_CABEZA. Misma familia que BLANCO y por la misma razon: sin
# exportarla, el tramo usaria su default y las tres condiciones correrian IGUAL sin avisar.
export PERDIDA_CABEZA="${PERDIDA_CABEZA:-bce}"
export P_VIEJA="${P_VIEJA:-0.35}"
export MEZCLA_PISO="${MEZCLA_PISO:-0.10}"
# El horizonte de la curva de lr NO se exportaba, asi que `tramo_abst.sh` usaba su default de 20000
# aunque el rotador corriera hasta otro paso. Con PASOS=26000 eso dejaba los ultimos 6000 pasos a la
# lr minima, que no es lo que la enmienda E-1 declaro. Cazado en el paso 1000 leyendo la config del
# primer checkpoint, no el log del rotador (D-1 del 22-ago).
export HORIZONTE="${HORIZONTE:-$PASOS}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/maxi/.venv-ligamento/bin/python}"   # 2026-08-25: lo usa completa() para leer el paso del checkpoint
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
  # 2026-08-25 · ANTES miraba el JSON de salidas DE HOY. Una unidad terminada en una corrida
  # ANTERIOR no tiene JSON hoy, asi que nunca se marcaba completa: el rotador la reelegia en cada
  # vuelta, subia su checkpoint, recibia "ya esta completo" y volvia a empezar. `ef3_s0` (completa
  # desde el 23-ago) dejo al rotador `ef` tres horas y media sin tocar s1 ni s2.
  # La fuente de verdad es el PASO DEL CHECKPOINT, que es ademas lo que mira el vigia por la D-1.
  local u; u="$(uni_de "$1")"
  local ck="$AQUI/ckpts/${u}.pkl"
  [ -f "$ck" ] || return 1
  local p
  p="$("$PY" -c "import pickle,sys;print(pickle.load(open(sys.argv[1],'rb')).get('paso') or 0)" "$ck" 2>/dev/null)"
  [ -n "$p" ] && [ "$p" -ge "$PASOS" ] 2>/dev/null
}

bloqueada() {
  local u; u="$(uni_de "$1")"
  local lk="$CKPTS/${u}.local.lock"
  [ -f "$lk" ] || return 1
  local pid; pid="$(cat "$lk" 2>/dev/null)"
  # `!= $$` (2026-08-27): el lock existe para que OTRO proceso no tome la unidad, pero el rotador
  # tambien lee el suyo. Sin esta condicion, marcarle su propia unidad lo deja esperando para
  # siempre a que se libere a si mismo — que es exactamente lo que paso hoy con b3_s2 en TPU.
  if [ -n "$pid" ] && [ "$pid" != "$$" ] && kill -0 "$pid" 2>/dev/null; then return 0; fi
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

echo "== rotador2 · familia '$PREFIJO' · p_nose $P_NOSE · mezcla $MEZCLA · piso $MEZCLA_PISO · p_vieja $P_VIEJA · unidades $UNIDADES · cuentas: ${CUENTAS[*]}"
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
    # 2026-08-23 · RESPALDO EN TPU. Medido hoy a las 15:20, con las 13 cuentas: T4 daba 503 en 12 de
    # 13, y `--tpu v5e1` daba 503 en las 5 que probe. Pero los dos aceleradores se racionan por
    # separado, asi que pedir el segundo cuando el primero falla duplica las chances por vuelta y no
    # cuesta nada cuando hay T4 (ni se llega a intentar). El derecho ya lo tenemos: lo que NO
    # tenemos es L4/G4/H100/A100/v6e1, que contestan «you may not have quota or entitlement».
    #
    # La TPU ya esta soportada aguas abajo: `tramo_abst.sh` acepta `TpuDevice` y la deteccion del
    # acelerador va en un subproceso justamente porque en TPU el kernel se quedaba con el chip
    # tomado (lo que costo las 5 TPU del 15-ago). Ese bug esta arreglado.
    ACC=""
    # 2026-08-27 · ACEL elige a CUAL se le pide primero, sin cambiar nada mas. `ACEL=tpu` invierte el
    # orden: TPU primero, T4 de respaldo. Pedido de Maxi hoy —«corre alguna de las pruebas en una tpu
    # que a esta hora hay»— y hace falta para lo que de verdad importa: con DOS rotadores en paralelo,
    # el que se lanza segundo NO debe competirle las T4 al que ya viene corriendo. Como los dos
    # aceleradores se racionan por separado, dos rotadores pidiendo cosas distintas se estorban poco;
    # dos pidiendo lo mismo se sacan las sesiones entre si.
    # El default `t4` deja el comportamiento anterior EXACTAMENTE igual: sin ACEL, esto es el mismo
    # if de antes.
    if [ "${ACEL:-t4}" = "tpu" ]; then
      PRIMERO=(--tpu v5e1); SEGUNDO=(--gpu T4); N1="TPU v5e1"; N2="T4"
    else
      PRIMERO=(--gpu T4); SEGUNDO=(--tpu v5e1); N1="T4"; N2="TPU v5e1"
    fi
    if timeout -k 30 420 "${CL[@]}" new -s "$SESION" "${PRIMERO[@]}" >/dev/null 2>&1; then
      ACC="$N1"
    elif [ "${TPU_RESPALDO:-1}" = "1" ] && \
         timeout -k 30 420 "${CL[@]}" new -s "$SESION" "${SEGUNDO[@]}" >/dev/null 2>&1; then
      ACC="$N2"
      echo "   (sin $N1 en $c, pero SI hubo $N2)"
    else
      echo "   503 en $c (ni T4 ni TPU) — SIGUIENTE CUENTA ya (sin esperar)"
      soltar_cuenta "$c"; continue
    fi
    HW="$(timeout -k 30 180 "${CL[@]}" status -s "$SESION" 2>&1 | tail -1)"
    echo "   >> OTORGADA en $c ($ACC): $HW"
    # SIN aviso por asignacion (2026-08-24). Esto mandaba un Telegram cada vez que una cuenta
    # otorgaba, y con 8 rotadores a la vez y el pool abriendose de golpe fueron cientos de mensajes
    # en una mañana. Queda en el log, que es donde se lo busca cuando importa; el estado agregado ya
    # lo manda `vigia_escalonado.sh` cada media hora con la foto de todas las unidades.

    for u in $FALTAN; do
      completa "$u" && continue
      echo "   -> tramo $u en $c (desde el paso $(paso_de "$u"))"
      "$AQUI/tramo_abst.sh" "$c" "$SESION" "$u" "$PASOS" "$TRAMO" "$CADA"
      # SIN aviso por tramo cerrado, por lo mismo: son varios por unidad por dia, y el latido del
      # vigia ya dice en que paso va cada una.
      echo "   tramo de $u cerrado en el paso $(paso_de "$u") de $PASOS"
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
