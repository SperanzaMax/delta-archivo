#!/usr/bin/env bash
# Worker de COLA: en cada vuelta toma la unidad MAS AVANZADA que falte y le suma un tramo.
#
#   Uso:  worker_cola.sh <CUENTA[,CUENTA...]> [pasos] [tramo] [cada]
#   Ej.:  worker_cola.sh G 12000 4000 1000          (una sola cuenta, como siempre)
#   Ej.:  worker_cola.sh A,C,D 12000 8000 1000      (rota a la siguiente tras 3 fallos seguidos)
#
# ROTACION DE CUENTA (2026-08-16, a pedido de Maxi). Insistir en la misma cuenta es la estrategia
# equivocada y ya estaba medido el 14-ago: a las 12:30 ninguna de J/D/E/I conseguia asignacion con
# 503 sostenido, mientras F/G/H seguian corriendo sin problema. Hoy paso la otra variante: la cuenta
# A consiguio la VM, la sesion murio con 404 a los pocos minutos, y el worker se quedaba
# reintentando con la MISMA cuenta cada 5 minutos indefinidamente.
#
# Por que rotar es seguro y no pierde trabajo: la continuidad NO depende de la cuenta sino del
# CHECKPOINT LOCAL. `tramo_colab.sh` sube `ckpts/<unidad>.pkl` a la VM antes de entrenar, asi que la
# cuenta nueva retoma exactamente en el paso que dejo la anterior. Cambiar de cuenta es transparente.
#
# Que cuenta como fallo, y es importante que sean los DOS casos:
#   (a) no consiguio acelerador  -> la cuenta esta sin cuota;
#   (b) consiguio VM pero el tramo NO avanzo el paso -> sesion perdida, 404, kernel trabado.
# Contar solo (a) dejaria al worker pegado en una cuenta que asigna pero cuyas sesiones se caen, que
# es exactamente lo que paso hoy.
#
# Por que cola y no una unidad fija por cuenta (2026-08-14):
# con GPU escasa, nueve workers atados a nueve unidades distintas terminan la jornada con NUEVE
# CORRIDAS A MEDIAS Y NINGUNA COMPLETA. Y una corrida a 4000 pasos no entra en la tabla, mientras
# que una a 12000 sí. La cola invierte la prioridad: se cierra lo que está más cerca de terminar
# antes de empezar nada nuevo, así cada ventana de GPU que se abre produce RESULTADO UTILIZABLE en
# vez de avance repartido.
#
# El reclamo evita que dos cuentas trabajen la misma unidad a la vez (seria trabajo duplicado y,
# peor, dos tramos escribiendo el mismo checkpoint). Vale 90 min: si el worker muere sin liberarlo,
# vence solo y la unidad vuelve a la cola.
set -uo pipefail

# La lista se recorre DE A UNA: en todo momento hay exactamente una cuenta activa. Rotar no es
# paralelizar —nunca hay dos cuentas trabajando a la vez— es pasar el testigo cuando la actual no
# sirve. Correr varias en paralelo es lo que el 15-ago hizo que cuatro cuentas tomaran n4_s0 juntas.
IFS=',' read -r -a CUENTAS <<< "${1:?falta la cuenta}"
TODAS=("${CUENTAS[@]}")        # lista original: las pausadas vuelven a ella cuando vence el TTL
CUENTA="${CUENTAS[0]}"
IDX=0
FALLOS=0
MAX_FALLOS="${MAX_FALLOS:-3}"
PASOS="${2:-12000}"
TRAMO="${3:-4000}"
CADA="${4:-1000}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAIMS="$AQUI/claims"; mkdir -p "$CLAIMS" "$AQUI/ckpts"
PY=/home/maxi/.venv-ligamento/bin/python
COLAB=/home/maxi/.venv-colab-cli/bin/colab
# SOLO T4 por defecto (2026-08-16). Antes pedia "T4 L4 A100" y se quedaba con el primero que
# contestara, sin registrar cual. Para un modelo de 3,5 MB a 0,22 s/paso una A100 no aporta NADA de
# velocidad util y consume creditos mucho mas rapido: aceptar la primera VM que conteste puede
# quemar la cuota de una cuenta en una sola corrida. L4/A100 quedan disponibles pero hay que
# pedirlos a proposito:  ACELERADORES="T4 L4" ./worker_cola.sh ...
ACELERADORES="${ACELERADORES:-T4}"
TPUS="${TPUS:-}"          # las TPU tambien quedan fuera por defecto: el 15-ago costaron 5 sesiones

# Familia de corridas. Se hereda a tramo_colab.sh por entorno.
#   PREFIJO=n · P_NOSE=0.0  -> campaña base: toda pregunta tiene respuesta en el archivo.
#   PREFIJO=x · P_NOSE=0.2  -> campaña de ABSTENCION: parte de las preguntas NO la tienen, que es
#                              la única forma de que `NOSE` pueda usarse. Con p_nose=0 la métrica
#                              sale NaN y la abstención no existe como opción medible.
# UNIDADES es la lista "nivel:semilla" que atiende esta cola, de mayor a menor prioridad al empatar.
PREFIJO="${PREFIJO:-n}"
P_NOSE="${P_NOSE:-0.0}"
UNIDADES="${UNIDADES:-1:0 1:1 1:2 2:0 2:1 2:2 3:0 3:1 3:2 4:0 4:1 4:2}"
export PREFIJO P_NOSE

usar_cuenta() {
  CUENTA="$1"
  if [ "$CUENTA" = "A" ]; then
    unset CLOUDSDK_CONFIG
    CL=( "$COLAB" --auth adc )
  else
    export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
    CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
  fi
}

# PRESUPUESTO DEL DIA (2026-08-16, observacion de Maxi). El recurso que se agota no es el tiempo:
# es la ASIGNACION. Entre errores y sesiones cortadas se pueden gastar los creditos de las trece
# cuentas en una jornada y quedarse sin nada para la noche. Dos topes, y los dos con memoria en
# disco para que sobrevivan a relanzar el worker:
#
#   TOPE_CUENTA  asignaciones que se le permiten a una cuenta EN EL DIA (default 4).
#   TOPE_DIA     asignaciones totales del dia entre todas las cuentas (default 10).
#
# Y la rotacion RETIRA la cuenta agotada en vez de volver a ella: con rotacion circular el worker
# vuelve a la cuenta quemada y sigue gastando toda la noche, que es exactamente lo que hay que
# evitar. Cuando no quedan cuentas con presupuesto, el worker PARA — no espera, no reintenta.
GASTO="$AQUI/gasto"; mkdir -p "$GASTO"
HOY="$(date +%Y%m%d)"
TOPE_CUENTA="${TOPE_CUENTA:-4}"
TOPE_DIA="${TOPE_DIA:-10}"

gasto_de()  { cat "$GASTO/${HOY}_$1" 2>/dev/null || echo 0; }
gasto_dia() { local t=0 f; for f in "$GASTO/${HOY}_"[A-Z]; do [ -f "$f" ] && t=$(( t + $(cat "$f") )); done; echo "$t"; }
anotar_gasto() { echo $(( $(gasto_de "$1") + 1 )) > "$GASTO/${HOY}_$1"; }

# APAGADO HASTA MAÑANA (2026-08-16, pedido de Maxi: «a medida que se agotan las apagás hasta
# mañana»). Retirar la cuenta solo de la lista en memoria no alcanza: si el worker se relanza, la
# cuenta quemada vuelve a probarse y se gasta de nuevo lo poco que le quede. La marca va a DISCO, y
# lleva la fecha en el nombre, asi que **manaña revive sola** sin tener que acordarse de nada.
#
# Se apaga por cualquiera de los dos motivos, porque los dos significan «esta cuenta ya no rinde
# hoy»: gasto el tope de asignaciones, o acumulo MAX_FALLOS seguidos sin producir un paso.
# DOS clases de retiro, porque los dos fallos NO cuestan lo mismo (2026-08-16, 2ª pasada):
#
#   · sin acelerador  -> NO se creo VM, NO se gasto credito. La cuenta puede tener T4 en 20 minutos.
#                        Apagarla hasta mañana es tirar una cuenta INTACTA: a 3 fallos por cuenta y
#                        ~20 min cada una, las 11 se «agotan» en 3 h sin haber gastado nada.
#                        -> PAUSA con TTL: vuelve a la rueda cuando pasa el rato.
#   · gasto del tope, o VM conseguida que no avanzo -> ahi SI se consumio la asignacion
#                        -> APAGADO hasta mañana.
#
# Pausar recicla, apagar descarta. La distincion importa justo el dia que hay que cerrar algo.
PAUSA_MIN="${PAUSA_MIN:-45}"

apagada() {
  [ -f "$GASTO/${HOY}_$1.off" ] && return 0
  local p="$GASTO/${HOY}_$1.pausa"
  [ -f "$p" ] || return 1
  local edad=$(( ($(date +%s) - $(stat -c %Y "$p")) / 60 ))
  if [ "$edad" -ge "$PAUSA_MIN" ]; then rm -f "$p"; return 1; fi
  return 0
}
apagar() { echo "$2" > "$GASTO/${HOY}_$1.off"; echo "   🔌 cuenta $1 APAGADA hasta mañana ($2)"; }
pausar() { echo "$2" > "$GASTO/${HOY}_$1.pausa"; echo "   ⏸ cuenta $1 EN PAUSA $PAUSA_MIN min ($2) — no gasto credito"; }
cuentas_vivas() { local c n=0; for c in "${CUENTAS[@]}"; do apagada "$c" || n=$(( n + 1 )); done; echo "$n"; }

# Retira la cuenta actual y toma la siguiente CON PRESUPUESTO. Devuelve 1 si no queda ninguna.
# El trabajo hecho no se toca: vive en el checkpoint local, que la cuenta nueva vuelve a subir.
#   rotar <motivo> [pausa]   — con «pausa» la cuenta se recicla en vez de descartarse.
rotar() {
  local previa="$CUENTA" motivo="${1:-$MAX_FALLOS fallos seguidos}" modo="${2:-off}"
  if [ "$modo" = "pausa" ]; then pausar "$previa" "$motivo"; else apagar "$previa" "$motivo"; fi
  unset 'CUENTAS[IDX]'; CUENTAS=("${CUENTAS[@]}")      # retirada: no se vuelve a esta cuenta hoy
  IDX=0
  while [ "${#CUENTAS[@]}" -gt 0 ]; do
    if apagada "${CUENTAS[0]}"; then
      echo "   (cuenta ${CUENTAS[0]} apagada hoy: $(cat "$GASTO/${HOY}_${CUENTAS[0]}.off"))"
      unset 'CUENTAS[0]'; CUENTAS=("${CUENTAS[@]}"); continue
    fi
    if [ "$(gasto_de "${CUENTAS[0]}")" -ge "$TOPE_CUENTA" ]; then
      apagar "${CUENTAS[0]}" "gasto $(gasto_de "${CUENTAS[0]}") asignaciones"
      unset 'CUENTAS[0]'; CUENTAS=("${CUENTAS[@]}"); continue
    fi
    usar_cuenta "${CUENTAS[0]}"; FALLOS=0
    echo "== ROTA de $previa a $CUENTA ($motivo) · quedan $(cuentas_vivas) cuentas vivas"
    echo "   (se retoma en el paso que dejo $previa: la continuidad la da el checkpoint local)"
    return 0
  done
  # Si lo unico que queda son PAUSAS, no se termina: se espera a que venzan. Solo se para de
  # verdad cuando todas estan apagadas por gasto real.
  local pausadas=0 c
  for c in "${TODAS[@]}"; do [ -f "$GASTO/${HOY}_$c.pausa" ] && pausadas=$(( pausadas + 1 )); done
  if [ "$pausadas" -gt 0 ]; then
    echo "== todas las cuentas en pausa ($pausadas); espera $PAUSA_MIN min y reintenta"
    sleep $(( PAUSA_MIN * 60 ))
    CUENTAS=("${TODAS[@]}"); IDX=0
    for c in "${CUENTAS[@]}"; do if ! apagada "$c"; then usar_cuenta "$c"; FALLOS=0; return 0; fi; done
  fi
  echo "== NO QUEDAN CUENTAS vivas hoy. El worker PARA; mañana revive todo solo."
  return 1
}

# Al arrancar se descartan las cuentas ya apagadas hoy. Esto es lo que hace que relanzar el worker
# —a mano, o despues de un corte— no vuelva a golpear lo que ya se quemo.
VIVAS=()
for c in "${CUENTAS[@]}"; do
  if apagada "$c"; then
    echo "   🔌 $c apagada hoy ($(cat "$GASTO/${HOY}_$c.off")): se saltea"
  elif [ "$(gasto_de "$c")" -ge "$TOPE_CUENTA" ]; then
    apagar "$c" "gasto $(gasto_de "$c") asignaciones"
  else
    VIVAS+=("$c")
  fi
done
CUENTAS=("${VIVAS[@]}")
if [ "${#CUENTAS[@]}" -eq 0 ]; then
  echo "== NO QUEDAN CUENTAS vivas hoy. Nada que hacer; mañana revive todo solo."; exit 0
fi
IDX=0
usar_cuenta "${CUENTAS[0]}"
echo "== cuentas disponibles hoy (${#CUENTAS[@]}): ${CUENTAS[*]}"

paso_de() {
  local f="$AQUI/ckpts/$1.pkl"
  [ -f "$f" ] || { echo 0; return; }
  "$PY" -c "import pickle;print(pickle.load(open('$f','rb'))['paso'])" 2>/dev/null || echo 0
}

# Lista TODAS las unidades incompletas y no reclamadas, de mas avanzada a menos. Empata a favor del
# nivel mas bajo. Devuelve la lista entera —no solo la mejor— porque quien llama tiene que poder
# bajar a la siguiente cuando pierde la carrera por la primera.
candidatas() {
  for ns in $UNIDADES; do
    local n="${ns%%:*}" s="${ns##*:}"
    local u="${PREFIJO}${n}_s${s}" cl="$CLAIMS/${PREFIJO}${n}_s${s}"
    local p; p="$(paso_de "$u")"
    [ "$p" -ge "$PASOS" ] && continue
    # reclamo vigente de otra cuenta -> saltear (90 min = 5400 s)
    if [ -f "$cl" ]; then
      local edad=$(( $(date +%s) - $(stat -c %Y "$cl") ))
      if [ "$edad" -lt 5400 ] && [ "$(cat "$cl")" != "$CUENTA" ]; then continue; fi
    fi
    printf '%s %s:%s\n' "$p" "$n" "$s"
  done | sort -k1,1nr -k2,2 | awk '{print $2}'
}

# Reclamo ATOMICO (2026-08-15, costo cuatro VMs en la primera tanda del dia).
#
# Antes el reclamo se escribia DESPUES de conseguir la VM, con el argumento de que reclamar primero
# bloquearia una unidad que quiza no se pueda trabajar. El razonamiento estaba al reves: entre elegir
# y tener la VM pasan minutos, asi que nueve workers que arrancan juntos eligen TODOS la misma unidad
# —la mas avanzada— y recien al final descubren que se pisaron. Pasó exacto: F, G, H y D tomaron
# n4_s0 a la vez, cuatro tramos apuntando al mismo checkpoint. El riesgo que se evitaba (una unidad
# reservada de mas) es barato y se resuelve liberando; el que se corria (perder el checkpoint) no.
#
# `set -o noclobber` hace el `>` atomico: si el archivo ya existe, falla en vez de sobrescribir. Es
# lo que convierte al reclamo en un candado real y no en un aviso.
reclamar() {
  local cl="$CLAIMS/$1"
  if [ -f "$cl" ]; then
    local edad=$(( $(date +%s) - $(stat -c %Y "$cl" 2>/dev/null || date +%s) ))
    # vencido o propio -> se puede retomar
    if [ "$edad" -ge 5400 ] || [ "$(cat "$cl" 2>/dev/null)" = "$CUENTA" ]; then rm -f "$cl"; else return 1; fi
  fi
  ( set -o noclobber; echo "$CUENTA" > "$cl" ) 2>/dev/null
}

echo "== worker de cola · cuenta $CUENTA · objetivo $PASOS pasos por unidad"
for vuelta in $(seq 1 "${VUELTAS:-200}"); do
  # Se reclama ANTES de pedir la VM, bajando por la lista hasta ganar una unidad para uno solo.
  U=""
  for cand in $(candidatas); do
    if reclamar "${PREFIJO}${cand%%:*}_s${cand##*:}"; then U="$cand"; break; fi
  done
  if [ -z "$U" ]; then
    if [ -z "$(candidatas)" ] && [ -z "$(ls -A "$CLAIMS" 2>/dev/null)" ]; then
      echo "== no queda nada por hacer"; break
    fi
    echo "-- v$vuelta: todo lo pendiente esta tomado por otra cuenta; espera 3 min"; sleep 180; continue
  fi
  N="${U%%:*}"; S="${U##*:}"

  # Los topes se miran ANTES de pedir la VM: una vez asignada, el credito ya se gasto.
  if [ "$(gasto_dia)" -ge "$TOPE_DIA" ]; then
    rm -f "$CLAIMS/${PREFIJO}${N}_s${S}"
    echo "== TOPE DEL DIA alcanzado ($(gasto_dia)/$TOPE_DIA asignaciones). El worker para."
    break
  fi
  if [ "$(gasto_de "$CUENTA")" -ge "$TOPE_CUENTA" ]; then
    rm -f "$CLAIMS/${PREFIJO}${N}_s${S}"
    echo "-- v$vuelta: $CUENTA agoto su presupuesto ($(gasto_de "$CUENTA")/$TOPE_CUENTA)"
    rotar "gasto $(gasto_de "$CUENTA") asignaciones" || break
    continue
  fi

  SES="q_${CUENTA,,}_${vuelta}"
  ASIGNO=0
  ACC_USADO=""
  for acc in $ACELERADORES; do
    if timeout 420 "${CL[@]}" new -s "$SES" --gpu "$acc" 2>&1 | grep -q "READY"; then
      ASIGNO=1; ACC_USADO="$acc"; break
    fi
  done
  if [ "$ASIGNO" = "0" ] && [ -n "${TPUS:-}" ]; then
    for tpu in $TPUS; do
      timeout 420 "${CL[@]}" new -s "$SES" --tpu "$tpu" 2>&1 | grep -q "READY" && { ASIGNO=1; ACC_USADO="TPU-$tpu"; break; }
    done
  fi
  if [ "$ASIGNO" = "0" ]; then
    # Sin VM no se trabaja: se devuelve la unidad a la cola para que otra cuenta la agarre.
    rm -f "$CLAIMS/${PREFIJO}${N}_s${S}"
    FALLOS=$(( FALLOS + 1 ))
    echo "-- v$vuelta: sin acelerador en $CUENTA (${PREFIJO}${N}_s${S} liberada) · fallo $FALLOS/$MAX_FALLOS"
    if [ "$FALLOS" -ge "$MAX_FALLOS" ]; then rotar "sin acelerador $MAX_FALLOS veces" pausa || break; continue; fi
    sleep 300; continue
  fi

  # La VM esta asignada: el credito ya se gasto, se anota pase lo que pase de aca en adelante.
  # Se registra TAMBIEN que acelerador toco: «gaste 3 asignaciones» y «gaste 1 A100» no son lo
  # mismo, y hasta hoy el log no lo distinguia ni siquiera para los dias que anduvieron bien.
  anotar_gasto "$CUENTA"
  echo "$(date +%H:%M) $CUENTA $ACC_USADO ${PREFIJO}${N}_s${S}" >> "$GASTO/${HOY}.acc"
  echo "   [presupuesto] $CUENTA $(gasto_de "$CUENTA")/$TOPE_CUENTA · dia $(gasto_dia)/$TOPE_DIA · acelerador $ACC_USADO"

  touch "$CLAIMS/${PREFIJO}${N}_s${S}"   # el TTL de 90 min cuenta desde que empieza el trabajo, no desde el pedido

  # El tramo se estira hasta CERRAR la unidad (2026-08-15). El recurso escaso no es el tiempo de VM
  # sino la ASIGNACION: una cuenta aguanta unas pocas antes de que Colab le empiece a contestar 503,
  # y hoy se estaba gastando una entera por cada 4000 pasos. Con el tramo pegado a lo que falta, una
  # sola asignación puede cerrar una unidad de punta a punta.
  # No agrega riesgo: el checkpoint se baja cada ~8 min, así que una VM que se cae cuesta 8 minutos
  # tanto en un tramo de 4000 como en uno de 12000. Y el presupuesto de polling de tramo_colab.sh
  # escala con el tramo, con lo que no corta antes de tiempo.
  P_ACT="$(paso_de "${PREFIJO}${N}_s${S}")"
  FALTA=$(( PASOS - P_ACT ))
  ESTE=$TRAMO
  [ "$FALTA" -lt "$ESTE" ] && ESTE=$FALTA
  echo "-- v$vuelta: acelerador OK · toma ${PREFIJO}${N}_s${S} (paso $P_ACT de $PASOS · este tramo +$ESTE)"
  "$AQUI/tramo_colab.sh" "$CUENTA" "$SES" "$U" "$PASOS" "$ESTE" "$CADA"
  timeout 180 "${CL[@]}" stop -s "$SES" >/dev/null 2>&1 || true
  rm -f "$CLAIMS/${PREFIJO}${N}_s${S}"

  P="$(paso_de "${PREFIJO}${N}_s${S}")"
  echo "-- v$vuelta cerrada: ${PREFIJO}${N}_s${S} en el paso $P de $PASOS"
  [ "$P" -ge "$PASOS" ] && echo "   ✅ ${PREFIJO}${N}_s${S} COMPLETA"

  # Caso (b): hubo VM pero el checkpoint NO avanzo -> sesion perdida (404), kernel trabado, o la VM
  # se cayo antes del primer volcado. Cuenta como fallo: si no, el worker se queda pegado en una
  # cuenta que asigna pero cuyas sesiones se mueren, que es lo que paso el 16-ago con la cuenta A.
  if [ "$P" -le "$P_ACT" ]; then
    FALLOS=$(( FALLOS + 1 ))
    echo "   ⚠ el tramo no avanzo (sigue en $P) · fallo $FALLOS/$MAX_FALLOS en $CUENTA"
    if [ "$FALLOS" -ge "$MAX_FALLOS" ]; then rotar "$MAX_FALLOS tramos sin avanzar" || break; fi
  else
    # Progreso real: la cuenta sirve, se le perdona lo anterior.
    [ "$FALLOS" -gt 0 ] && echo "   (avanzo $(( P - P_ACT )) pasos: contador de fallos de $CUENTA a cero)"
    FALLOS=0
  fi
done
echo "== worker de cola $CUENTA fin"
