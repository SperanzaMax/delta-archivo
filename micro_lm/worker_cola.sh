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

# Elige la unidad incompleta, no reclamada, con MAS pasos hechos. Empata a favor del nivel mas bajo.
elegir() {
  local mejor="" mejorp=-1
  for n in 1 2 3 4; do for s in 0 1 2; do
    local u="n${n}_s${s}" cl="$CLAIMS/n${n}_s${s}"
    local p; p="$(paso_de "$u")"
    [ "$p" -ge "$PASOS" ] && continue
    # reclamo vigente de otra cuenta -> saltear (90 min = 5400 s)
    if [ -f "$cl" ]; then
      local edad=$(( $(date +%s) - $(stat -c %Y "$cl") ))
      if [ "$edad" -lt 5400 ] && [ "$(cat "$cl")" != "$CUENTA" ]; then continue; fi
    fi
    if [ "$p" -gt "$mejorp" ]; then mejorp="$p"; mejor="$n:$s"; fi
  done; done
  echo "$mejor"
}

echo "== worker de cola · cuenta $CUENTA · objetivo $PASOS pasos por unidad"
for vuelta in $(seq 1 "${VUELTAS:-200}"); do
  U="$(elegir)"
  if [ -z "$U" ]; then echo "== no queda nada por hacer"; break; fi
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
    echo "-- v$vuelta: sin acelerador; espera 5 min"; sleep 300; continue
  fi

  # Se reclama DESPUES de tener la VM: reclamar antes bloquearia la unidad sin poder trabajarla.
  echo "$CUENTA" > "$CLAIMS/n${N}_s${S}"
  echo "-- v$vuelta: acelerador OK · toma n${N}_s${S} (paso $(paso_de "n${N}_s${S}") de $PASOS)"
  "$AQUI/tramo_colab.sh" "$CUENTA" "$SES" "$U" "$PASOS" "$TRAMO" "$CADA"
  timeout 180 "${CL[@]}" stop -s "$SES" >/dev/null 2>&1 || true
  rm -f "$CLAIMS/n${N}_s${S}"

  P="$(paso_de "n${N}_s${S}")"
  echo "-- v$vuelta cerrada: n${N}_s${S} en el paso $P de $PASOS"
  [ "$P" -ge "$PASOS" ] && echo "   ✅ n${N}_s${S} COMPLETA"
done
echo "== worker de cola $CUENTA fin"
