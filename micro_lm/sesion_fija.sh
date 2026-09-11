#!/usr/bin/env bash
# Corre UNA unidad entera, por tramos, en una sesion de Colab que YA EXISTE (2026-09-11).
#
#   Uso:  sesion_fija.sh <CUENTA> <sesion> <nivel:semilla> <pasos> <tramo> <cada>
#
# Existe porque a las 08:50 del 11-sep la campania en frio se paro con seis T4 ya otorgadas y
# vivas: pedirlas de nuevo con el rotador es lo escaso, asi que el plan B las reusa tal cual.
# Encadena `tramo_abst.sh` hasta que el checkpoint llegue a <pasos>, y si un tramo no avanza el
# paso (sesion perdida, 404) se rinde y lo dice: ahi entra el rotador normal desde el checkpoint.
# Al terminar PARA la sesion y avisa por Telegram.
set -uo pipefail
CUENTA="${1:?}"; SESION="${2:?}"; UNIDAD="${3:?}"; PASOS="${4:?}"; TRAMO="${5:?}"; CADA="${6:?}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$AQUI/tg_token.sh" 2>/dev/null || true
COLAB=/home/maxi/.venv-colab-cli/bin/colab
NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"
UNI="${PREFIJO:-n}${NIVEL}_s${SEM}"
CK="$AQUI/ckpts/${UNI}.pkl"
PY=/home/maxi/.venv-ligamento/bin/python
paso_de() { [ -f "$CK" ] && $PY -c "import pickle,sys;print(pickle.load(open(sys.argv[1],'rb')).get('paso',0))" "$CK" 2>/dev/null || echo 0; }
mandar() { [ -n "${TOKEN:-}" ] && curl -s -m 30 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null; true; }
if [ "$CUENTA" = "A" ]; then CL=( "$COLAB" --auth adc ); else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"; CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" ); fi
echo "== sesion fija · $UNI en $CUENTA/$SESION · desde el paso $(paso_de) hasta $PASOS"
for _ in $(seq 1 20); do
  P0="$(paso_de)"
  [ "$P0" -ge "$PASOS" ] && break
  "$AQUI/tramo_abst.sh" "$CUENTA" "$SESION" "$UNIDAD" "$PASOS" "$TRAMO" "$CADA"
  P1="$(paso_de)"
  if [ "$P1" -le "$P0" ]; then
    echo "!! $UNI no avanzo ($P0 -> $P1): sesion perdida o kernel trabado. Me rindo; relanzar con el rotador."
    mandar "🔴 $UNI en $CUENTA no avanzo del paso $P0. Relanzar con rotar_abst3.sh."
    exit 1
  fi
done
echo "== $UNI completa ($(paso_de) pasos); se para la sesion $SESION"
timeout -k 30 120 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1
mandar "✅ $UNI llego al paso $(paso_de) en $CUENTA."
