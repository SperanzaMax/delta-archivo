#!/usr/bin/env bash
# EL BANCO DE POOL FRESCO EN UNA T4 · 2026-09-11
#   Uso:  banco_fresco_colab.sh <CUENTA> <ckpt.pkl> [<ckpt.pkl> ...]
# Corre `dilucion.py` (DIST=real TURNOS=viejo PERT=1, XS=0,360,3240, NMUE=256) sobre los checkpoints
# dados en una sesion nueva de Colab y se trae los JSON a controles_<fecha>/. En la PC tarda 12 min
# por checkpoint a un nucleo (la regla del 80 %); en la T4, segundos.
set -uo pipefail
CUENTA="${1:?falta la cuenta}"; shift
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
SAL="$AQUI/controles_$(date +%Y%m%d)"; mkdir -p "$SAL"
if [ "$CUENTA" = "A" ]; then CL=( "$COLAB" --auth adc ); else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"; CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" ); fi
SESION="banco_$(date +%H%M)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
timeout -k 30 420 "${CL[@]}" new -s "$SESION" --gpu T4 >/dev/null 2>&1 || { echo "sin T4 en $CUENTA"; exit 1; }
tar czf "$TMP/micro.tgz" -C "$AQUI" idioma.py datos.py modelo.py conf_ckpt.py dilucion.py
timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$TMP/micro.tgz" /content/micro.tgz || exit 1
NOMBRES=()
for ck in "$@"; do
  b="$(basename "$ck")"; NOMBRES+=("$b")
  timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$ck" "/content/$b" || exit 1
done
cat > "$TMP/correr.py" <<PY
import os, subprocess, sys, json
os.makedirs('/content/micro/ckpts', exist_ok=True)
subprocess.run('tar xzf /content/micro.tgz -C /content/micro', shell=True, check=True)
nombres = "${NOMBRES[*]}".split()
for b in nombres:
    os.replace('/content/' + b, '/content/micro/ckpts/' + b)
env = dict(os.environ, DIST='real', TURNOS='viejo', PERT='${PERT:-1}', XS='${XS:-0,360,3240}', NMUE='${NMUE:-256}', POOL='${POOL:-4096}')
r = subprocess.run([sys.executable, 'dilucion.py'] + ['ckpts/' + b for b in nombres], cwd='/content/micro', env=env, capture_output=True, text=True)
print(r.stdout[-6000:]); print(r.stderr[-2000:], file=sys.stderr)
import glob
print('@@JSON@@ ' + json.dumps(json.load(open(sorted(glob.glob('/content/micro/dilucion_real_viejo*.json'), key=os.path.getmtime)[-1]))))
PY
OUT="$(timeout -k 30 900 "${CL[@]}" exec -s "$SESION" --timeout 840 -f "$TMP/correr.py" 2>&1)"
printf '%s\n' "$OUT" | grep -avE "^@@JSON@@" | tail -30
printf '%s\n' "$OUT" | grep -a "^@@JSON@@ " | sed 's/^@@JSON@@ //' > "$SAL/banco_fresco_pert${PERT:-1}_${ETIQUETA:-x}_$(date +%H%M).json"
echo "== json en $SAL/banco_fresco_pert${PERT:-1}_${ETIQUETA:-x}_$(date +%H%M).json"
timeout -k 30 120 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1
