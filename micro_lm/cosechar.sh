#!/usr/bin/env bash
# Recolecta los JSON de una sesion de Colab que YA ESTA corriendo. No crea, no lanza, no reintenta.
#
#   Uso:  cosechar.sh <CUENTA> <sesion> [minutos]
#   Ej.:  cosechar.sh H micro_h_3 150
#
# Por que separado del lanzador (2026-08-14): mezclar «crear sesion + lanzar + pollear» en un solo
# proceso hizo que cada tropiezo de Colab dejara el proceso colgado y, peor, que al reintentar
# quedaran dos procesos `colab` sobre la misma cuenta. El cosechador hace UNA cosa: un `exec` corto
# con timeout, cada N minutos, y escribe lo que traiga. Si el exec falla, no pasa nada: al siguiente
# tick se vuelve a intentar.
set -uo pipefail

CUENTA="${1:?falta la cuenta}"
SESION="${2:?falta la sesion}"
MINUTOS="${3:-150}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
mkdir -p "$SALIDA"

if [ "$CUENTA" = "A" ]; then
  CL=( "$COLAB" --auth adc )
else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
  CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
fi

VER="$(mktemp)"; trap 'rm -f "$VER"' EXIT
cat > "$VER" <<'PY'
import json, os
try:
    pid = int(open('/content/micro.pid').read()); print('VIVO=', os.path.exists('/proc/%d' % pid))
except Exception as e:
    print('VIVO= ?', e)
try:
    print('ULTIMO=', [l for l in open('/content/micro.log', errors='ignore') if l.strip()][-1].strip())
except Exception:
    pass
for f in sorted(os.listdir('/content/salidas')):
    if f.endswith('.json'):
        print('@@JSON@@', f, json.dumps(json.load(open('/content/salidas/' + f))))
PY

echo "== cosechando $CUENTA/$SESION cada 3 min durante ~$MINUTOS min"
for _ in $(seq 1 $(( MINUTOS / 3 ))); do
  OUT="$(timeout 200 "${CL[@]}" exec -s "$SESION" -f "$VER" 2>&1 || true)"
  { printf '%s\n' "$OUT" | grep '^@@JSON@@ ' || true; } | while read -r _ nombre resto; do
    printf '%s' "$resto" > "$SALIDA/$nombre"
  done
  { printf '%s\n' "$OUT" | grep -E "VIVO=|ULTIMO=" || true; } | tr '\n' ' '; echo
  if printf '%s' "$OUT" | grep -qE "not found|appears to be lost"; then
    echo "== la sesion ya no existe; fin"; break
  fi
  if printf '%s' "$OUT" | grep -q "VIVO= False"; then
    echo "== el runner termino; ultima cosecha hecha"; break
  fi
  sleep 180
done
echo "== cosecha terminada para $CUENTA/$SESION"
