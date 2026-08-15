#!/usr/bin/env bash
# Lleva UNA unidad (nivel:semilla) hasta el final, tramo a tramo, sobreviviendo a caidas de VM.
#
#   Uso:  worker.sh <CUENTA> <nivel:semilla> [pasos] [tramo] [cada]
#   Ej.:  worker.sh G 4:0 20000 4000 1000
#
# El bucle es: conseguir acelerador -> correr un tramo -> parar la sesion -> repetir hasta que el
# checkpoint llegue a `pasos`. Cada tramo arranca donde quedo el anterior, asi que una VM que se cae
# cuesta como mucho un intervalo de evaluacion, no la corrida.
#
# Se para la sesion DESPUES de cada tramo a proposito: una VM viva sin nadie usandola quema cuota, y
# ademas Colab parece cortar las sesiones largas. Tramos cortos con sesiones nuevas resultaron mas
# confiables que una sesion larga (el 14-ago: ocho VMs perdidas en una hora con sesiones largas).
set -uo pipefail

CUENTA="${1:?falta la cuenta}"
UNIDAD="${2:?falta nivel:semilla}"
PASOS="${3:-20000}"
TRAMO="${4:-4000}"
CADA="${5:-1000}"
ACELERADORES="${ACELERADORES:-T4 L4 A100}"

NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CK="$AQUI/ckpts/n${NIVEL}_s${SEM}.pkl"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
PY=/home/maxi/.venv-ligamento/bin/python

if [ "$CUENTA" = "A" ]; then
  CL=( "$COLAB" --auth adc )
else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
  CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
fi

paso_actual() {
  [ -f "$CK" ] || { echo 0; return; }
  "$PY" -c "import pickle;print(pickle.load(open('$CK','rb'))['paso'])" 2>/dev/null || echo 0
}

# VUELTAS alto por defecto: la cuota de GPU de Colab se libera sola cada tanto, y el 14-ago las
# nueve cuentas quedaron secas a la vez durante horas. Un worker que se rinde a la hora no sirve;
# lo que sirve es que siga golpeando la puerta toda la noche y arranque solo cuando se abra.
VUELTAS="${VUELTAS:-200}"
echo "== worker $CUENTA · n$NIVEL s$SEM · desde el paso $(paso_actual) hasta $PASOS"
for vuelta in $(seq 1 "$VUELTAS"); do
  P="$(paso_actual)"
  if [ "$P" -ge "$PASOS" ]; then
    echo "== n$NIVEL s$SEM COMPLETA ($P pasos)"; break
  fi

  SES="w_${CUENTA,,}_${NIVEL}${SEM}_${vuelta}"
  ASIGNO=0
  for acc in $ACELERADORES; do
    if timeout 420 "${CL[@]}" new -s "$SES" --gpu "$acc" 2>&1 | grep -q "READY"; then
      echo "-- vuelta $vuelta: $acc asignado ($SES), desde el paso $P"; ASIGNO=1; break
    fi
  done
  # La TPU va AL FINAL de la cascada, no por lenta de por sí sino porque este modelo es su peor
  # caso: la regla delta es un `scan` SECUENCIAL de 96 pasos sobre matrices de 128x128, o sea nada
  # que paralelizar y matmuls minúsculos para un chip pensado para lo contrario. Se la intenta
  # igual porque su cuota es SEPARADA de la de GPU: el 14-ago las nueve cuentas quedaron sin GPU y
  # una TPU lenta hubiera sido mejor que nada. NO está verificado que el modelo corra en TPU —
  # si sale, lo primero que hay que mirar es el s/paso contra los 0,46 de la T4.
  if [ "$ASIGNO" = "0" ]; then
    for tpu in ${TPUS:-v5e1}; do
      if timeout 420 "${CL[@]}" new -s "$SES" --tpu "$tpu" 2>&1 | grep -q "READY"; then
        echo "-- vuelta $vuelta: TPU $tpu asignada ($SES), desde el paso $P — VERIFICAR s/paso"
        ASIGNO=1; break
      fi
    done
  fi
  if [ "$ASIGNO" = "0" ]; then
    echo "-- vuelta $vuelta: sin acelerador en $CUENTA; espera 5 min"
    sleep 300; continue
  fi

  "$AQUI/tramo_colab.sh" "$CUENTA" "$SES" "$UNIDAD" "$PASOS" "$TRAMO" "$CADA"
  timeout 180 "${CL[@]}" stop -s "$SES" >/dev/null 2>&1 || true

  NUEVO="$(paso_actual)"
  echo "-- vuelta $vuelta cerrada: $P -> $NUEVO"
  if [ "$NUEVO" -le "$P" ]; then
    echo "   (no avanzo nada; espera 3 min antes de reintentar)"
    sleep 180
  fi
done
echo "== worker $CUENTA fin · n$NIVEL s$SEM en el paso $(paso_actual) de $PASOS"
