#!/usr/bin/env bash
# LA PELICULA LARGA · corre `pelicula.py` en una T4 y se trae el JSON cada tanto. 2026-09-05
#
#   Uso:  pelicula_colab.sh <CUENTA> <pasos> <cada> [sesion]
#   Ej.:  pelicula_colab.sh A 26000 500
#
# Por que no reusa `tramo_abst.sh`: aquel corre `entrenar.py` y baja un checkpoint; esto corre otro
# script y baja un JSON que CRECE (pelicula.py lo reescribe entero en cada cuadro). Se lo baja en
# cada vuelta de polling, asi que una sesion que se muere a los 40 minutos deja una pelicula de 40
# minutos en vez de nada.
#
# 26.000 pasos es el presupuesto de las campanias reales (kq3_sX), o sea que la pelicula termina
# donde termina el modelo del que ya tenemos la referencia.
set -uo pipefail

CUENTA="${1:?falta la cuenta}"
PASOS="${2:-26000}"
CADA="${3:-500}"
SESION="${4:-peli_$(date +%H%M)}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
DEST="$AQUI/pelicula_26000.json"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT

if [ "$CUENTA" = "A" ]; then
  CL=( "$COLAB" --auth adc )
else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
  CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
fi

echo "== pelicula · cuenta $CUENTA · sesion $SESION · $PASOS pasos · cuadro cada $CADA"
if ! timeout -k 30 420 "${CL[@]}" new -s "$SESION" --gpu T4 >/dev/null 2>&1; then
  echo "   503: sin T4 en $CUENTA"; exit 1
fi
echo "   >> $(timeout -k 30 180 "${CL[@]}" status -s "$SESION" 2>&1 | tail -1)"

tar czf "$TMP/micro.tgz" -C "$AQUI" idioma.py datos.py modelo.py entrenar.py pelicula.py
timeout 300 "${CL[@]}" upload -s "$SESION" "$TMP/micro.tgz" /content/micro.tgz || exit 1

cat > "$TMP/lanzar.py" <<PY
import subprocess, sys, os
os.makedirs('/content/micro', exist_ok=True)
subprocess.run(['tar','xzf','/content/micro.tgz','-C','/content/micro'], check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','optax'], check=False)
det = subprocess.run([sys.executable,'-c','import jax;print(jax.devices())'],
                     capture_output=True, text=True, cwd='/content/micro')
print(det.stdout.strip(), flush=True)
assert 'CudaDevice' in det.stdout, 'NO hay acelerador'
cmd = [sys.executable, '-u', 'pelicula.py', '--pasos', '$PASOS', '--cada', '$CADA',
       '--nivel', '3', '--d', '128', '--capas', '4', '--kernel-q', '5', '--donde', 'lat2',
       '--batch', '32', '--semilla', '0', '--salida', '/content/pelicula.json']
log = open('/content/peli.log','w')
p = subprocess.Popen(cmd, cwd='/content/micro', stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
open('/content/peli.pid','w').write(str(p.pid))
print('lanzado pid', p.pid, flush=True)
PY
timeout -k 30 300 "${CL[@]}" exec -s "$SESION" --timeout 240 -f "$TMP/lanzar.py" || exit 1

cat > "$TMP/ver.py" <<'PY'
import os, subprocess
pid = int(open('/content/peli.pid').read())
vivo = subprocess.run(['kill','-0',str(pid)], capture_output=True).returncode == 0
ult = ''
if os.path.exists('/content/peli.log'):
    ls = [l for l in open('/content/peli.log').read().splitlines() if l.strip()]
    ult = ls[-1] if ls else ''
print('VIVO=', vivo, 'ULTIMO=', ult, flush=True)
PY

echo "== polling cada 5 min; el JSON se baja en cada vuelta"
for i in $(seq 1 40); do
  sleep 300
  EST="$(timeout -k 30 240 "${CL[@]}" exec -s "$SESION" --timeout 180 -f "$TMP/ver.py" 2>&1 | grep VIVO=)"
  echo "   [$i] $EST"
  timeout -k 30 300 "${CL[@]}" download -s "$SESION" /content/pelicula.json "$DEST" >/dev/null 2>&1 \
    && echo "        bajado ($(du -h "$DEST" 2>/dev/null | cut -f1))"
  case "$EST" in *"VIVO= False"*) echo "== proceso terminado"; break;; esac
done

timeout -k 30 300 "${CL[@]}" download -s "$SESION" /content/pelicula.json "$DEST" >/dev/null 2>&1
timeout -k 30 180 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1 || true
echo "== fin · $DEST ($(du -h "$DEST" 2>/dev/null | cut -f1))"
