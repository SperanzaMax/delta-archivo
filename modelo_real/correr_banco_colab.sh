#!/usr/bin/env bash
# El banco de TinyLlama escalado, en una cuenta del pool. · 2026-09-09
#
#   Uso:  correr_banco_colab.sh <CUENTA> <UNIDAD> <semilla> [pasos] [extra...]
#   Ej.:  correr_banco_colab.sh C base 0 1200
#         correr_banco_colab.sh D sinord 0 1200 --sin-ord
#
# Adaptado de `correr_real_colab.sh`. Dos diferencias: NO instala mambapy (TinyLlama es un Llama
# comun, no un SSM) y sube el pool de vocabulario junto al script.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
CUENTA="${1:?falta la cuenta, p.ej. C}"
UNI_N="${2:?falta el nombre de la unidad}"
SEM="${3:?falta la semilla}"
PASOS="${4:-1200}"
shift 4 2>/dev/null || shift $#
EXTRA=("$@")

UNI="banco_${UNI_N}_s${SEM}"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
POOL="$HOME/.colab-pool"
. ../micro_lm/tg_token.sh
mandar(){ curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1; }

if [ "$CUENTA" = "A" ]; then CL=( "$COLAB" --auth adc ); unset CLOUDSDK_CONFIG
else export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"; CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" ); fi

lk="$POOL/en_uso_$CUENTA"
if [ -f "$lk" ] && kill -0 "$(cat "$lk" 2>/dev/null)" 2>/dev/null; then
  echo "cuenta $CUENTA ocupada por el pid $(cat "$lk")"; exit 1
fi
echo $$ > "$lk"; trap 'rm -f "$lk"' EXIT

VIVAS="$(timeout -k 20 180 "${CL[@]}" sessions 2>/dev/null | grep -iE "T4|L4|GPU|TPU" || true)"
SESION="$(echo "$VIVAS" | head -1 | sed -n 's/^\[\([^]]*\)\].*/\1/p')"
if [ -n "$SESION" ]; then
  echo "== $UNI en la cuenta $CUENTA · REUSA la sesion $SESION · $PASOS pasos"
else
  SESION="banco_${CUENTA,,}_$(date +%H%M)"
  echo "== $UNI en la cuenta $CUENTA · sesion NUEVA $SESION · $PASOS pasos"
  OUT="$(timeout -k 30 600 "${CL[@]}" new -s "$SESION" --gpu T4 2>&1 | tail -3)"
  echo "$OUT"
  echo "$OUT" | grep -qi "READY" || { echo "sin sesion en $CUENTA"; exit 1; }
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; rm -f "$lk"' EXIT
tar czf "$TMP/banco.tgz" banco_escalado.py pool_tinyllama.json
timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$TMP/banco.tgz" /content/banco.tgz || exit 1

EXTRA_PY="$(printf "'%s', " "${EXTRA[@]:-}" | sed "s/'', $//")"
cat > "$TMP/lanzar.py" <<PY
import os, subprocess, sys
os.makedirs('/content/banco', exist_ok=True)
subprocess.run('tar xzf /content/banco.tgz -C /content/banco', shell=True, check=True)
det = subprocess.run([sys.executable, '-c',
    'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'],
    capture_output=True, text=True)
print('entorno', det.stdout.strip(), det.stderr.strip()[:200], flush=True)
assert 'True' in det.stdout, 'NO hay GPU'
cmd = [sys.executable, '-u', 'banco_escalado.py', '--semilla', '$SEM', '--pasos', '$PASOS',
       '--cada', '25', '--salida', '/content/${UNI}.json', $EXTRA_PY]
cmd = [c for c in cmd if c]
print('cmd', ' '.join(cmd), flush=True)
log = open('/content/banco.log', 'w')
p = subprocess.Popen(cmd, cwd='/content/banco', stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
open('/content/banco.pid', 'w').write(str(p.pid))
print('lanzado pid', p.pid, flush=True)
PY
timeout -k 30 900 "${CL[@]}" exec -s "$SESION" --timeout 600 -f "$TMP/lanzar.py" || exit 1

cat > "$TMP/ver.py" <<'PY'
import os
try:
    pid = int(open('/content/banco.pid').read())
    vivo = os.path.exists('/proc/%d' % pid)
    if vivo:
        try:
            if open('/proc/%d/stat' % pid).read().split(')')[-1].split()[0] == 'Z':
                vivo = False
        except Exception:
            pass
    print('VIVO=', vivo)
except Exception as e:
    print('VIVO= ?', e)
try:
    print('ULTIMO=', [l for l in open('/content/banco.log') if l.strip()][-1].strip()[:220])
except Exception:
    print('ULTIMO= (sin log)')
PY

for i in $(seq 1 60); do
  sleep 120
  OUT="$(timeout -k 20 180 "${CL[@]}" exec -s "$SESION" --timeout 120 -f "$TMP/ver.py" 2>&1 | tail -2)"
  echo "$OUT"
  echo "$OUT" | grep -q "VIVO= False" && break
done

timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/${UNI}.json" "$AQUI/${UNI}.json" 2>&1 | tail -1
timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/banco.log" "$AQUI/${UNI}.log" 2>&1 | tail -1
timeout -k 30 120 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1
if [ -f "$AQUI/${UNI}.json" ]; then
  echo "== $UNI LISTO"
  mandar "🧪 banco escalado · $UNI cerró
$(tail -8 "$AQUI/${UNI}.log" 2>/dev/null)"
else
  echo "== $UNI SIN RESULTADO"
  mandar "⚠️ banco escalado · $UNI SIN RESULTADO en la cuenta $CUENTA"
fi
