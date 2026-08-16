#!/usr/bin/env bash
# Worker de COLA: en cada vuelta toma la unidad MAS AVANZADA que falte y le suma un tramo.
#
#   Uso:  worker_cola.sh <CUENTA> [pasos] [tramo] [cada]
#   Ej.:  worker_cola.sh G 12000 4000 1000
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

CUENTA="${1:?falta la cuenta}"
PASOS="${2:-12000}"
TRAMO="${3:-4000}"
CADA="${4:-1000}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLAIMS="$AQUI/claims"; mkdir -p "$CLAIMS" "$AQUI/ckpts"
PY=/home/maxi/.venv-ligamento/bin/python
COLAB=/home/maxi/.venv-colab-cli/bin/colab
ACELERADORES="${ACELERADORES:-T4 L4 A100}"

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

if [ "$CUENTA" = "A" ]; then
  CL=( "$COLAB" --auth adc )
else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
  CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
fi

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

  SES="q_${CUENTA,,}_${vuelta}"
  ASIGNO=0
  for acc in $ACELERADORES; do
    if timeout 420 "${CL[@]}" new -s "$SES" --gpu "$acc" 2>&1 | grep -q "READY"; then
      ASIGNO=1; break
    fi
  done
  if [ "$ASIGNO" = "0" ]; then
    for tpu in ${TPUS:-v5e1}; do
      timeout 420 "${CL[@]}" new -s "$SES" --tpu "$tpu" 2>&1 | grep -q "READY" && { ASIGNO=1; break; }
    done
  fi
  if [ "$ASIGNO" = "0" ]; then
    # Sin VM no se trabaja: se devuelve la unidad a la cola para que otra cuenta la agarre.
    rm -f "$CLAIMS/${PREFIJO}${N}_s${S}"
    echo "-- v$vuelta: sin acelerador (${PREFIJO}${N}_s${S} liberada); espera 5 min"; sleep 300; continue
  fi

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
done
echo "== worker de cola $CUENTA fin"
