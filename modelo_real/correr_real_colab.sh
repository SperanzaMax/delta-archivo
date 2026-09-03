#!/usr/bin/env bash
# Corre UNA unidad del experimento en modelo real, en una cuenta del pool.
#
#   Uso:  correr_real_colab.sh <CUENTA> <condicion> <semilla> [pasos]
#   Ej.:  correr_real_colab.sh H dos 0 3000
#
# A diferencia de la campania del micro-LM, esto NO va por tramos: el fine-tune entra en una sola
# sesion. Si algun dia no entra, hay que partirlo, pero primero hay que medir cuanto tarda de verdad.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
CUENTA="${1:?falta la cuenta, p.ej. H}"
COND="${2:?falta la condicion: una | dos | ciega}"
SEM="${3:?falta la semilla}"
PASOS="${4:-1200}"
NH="${NH:-16}"     # hechos en el contexto; con 4 la tarea SATURA, medido el 2-sep
MODELO="${MODELO:-state-spaces/mamba-130m-hf}"
# 2026-09-02, MEDIDO y no estimado: con mamba-370m y SIN los kernels `mamba-ssm`/`causal-conv1d`
# (que Colab no trae) HF cae al camino secuencial en Python y el paso cuesta 9,7 s. 3000 pasos serian
# 8 h y no entran en una sesion. El 130m es el mismo modelo real, misma arquitectura y MISMO alcance
# medido de 2 tokens, y ademas comparte tokenizer exacto, asi que las distancias no cambian.
UNI="real_${COND}_s${SEM}"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
POOL="$HOME/.colab-pool"
. ../micro_lm/tg_token.sh
mandar(){ curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1; }

if [ "$CUENTA" = "A" ]; then CL=( "$COLAB" --auth adc ); unset CLOUDSDK_CONFIG
else export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"; CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" ); fi

# lock por cuenta, misma razon que en el rotador del micro-LM: dos procesos sobre la misma cuenta
# se pisan el sessions.json y dejan la VM inalcanzable
lk="$POOL/en_uso_$CUENTA"
if [ -f "$lk" ] && kill -0 "$(cat "$lk" 2>/dev/null)" 2>/dev/null; then
  echo "cuenta $CUENTA ocupada por el pid $(cat "$lk")"; exit 1
fi
echo $$ > "$lk"; trap 'rm -f "$lk"' EXIT

# REUSAR antes que crear. Colab tira `TooManyAssignmentsError` si la cuenta ya tiene una VM
# asignada, y un intento fallido deja la sesion viva: pedir otra falla para siempre hasta pararla.
# Costo un lanzamiento hoy. Se busca primero una sesion con acelerador y recien si no hay se crea.
VIVAS="$(timeout -k 20 180 "${CL[@]}" sessions 2>/dev/null | grep -iE "T4|L4|GPU|TPU" || true)"
SESION="$(echo "$VIVAS" | head -1 | sed -n 's/^\[\([^]]*\)\].*/\1/p')"
if [ -n "$SESION" ]; then
  echo "== $UNI en la cuenta $CUENTA · REUSA la sesion $SESION · $PASOS pasos"
else
  SESION="real_${CUENTA,,}_$(date +%H%M)"
  echo "== $UNI en la cuenta $CUENTA · sesion NUEVA $SESION · $PASOS pasos"
  # `new`, no `create`, y la T4 se pide por nombre: sin --gpu sale una VM de CPU y el assert de
  # GPU de `lanzar.py` aborta despues de haber gastado la asignacion.
  OUT="$(timeout -k 30 600 "${CL[@]}" new -s "$SESION" --gpu T4 2>&1 | tail -3)"
  echo "$OUT"
  echo "$OUT" | grep -qi "READY" || { echo "sin sesion en $CUENTA"; exit 1; }
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; rm -f "$lk"' EXIT
tar czf "$TMP/real.tgz" tarea_real.py entrenar_real.py vocabulario.json
timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$TMP/real.tgz" /content/real.tgz || exit 1

cat > "$TMP/lanzar.py" <<PY
import os, subprocess, sys
os.makedirs('/content/real', exist_ok=True)
subprocess.run('tar xzf /content/real.tgz -C /content/real', shell=True, check=True)
det = subprocess.run([sys.executable, '-c',
    'import torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.is_available())'],
    capture_output=True, text=True)
print('entorno', det.stdout.strip(), det.stderr.strip()[:200], flush=True)
assert 'True' in det.stdout, 'NO hay GPU'
# 2026-09-03: mambapy es pip PURO (40 kB, sin compilar CUDA) y da el scan paralelo. Es lo que
# destraba las dos paredes del 2-sep. Si no instala, la corrida NO sigue: con el camino secuencial
# no entra en una sesion y gastariamos la asignacion al pedo.
ins = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mambapy'],
                     capture_output=True, text=True)
chk = subprocess.run([sys.executable, '-c', 'from mambapy.pscan import pscan; print("pscan OK")'],
                     capture_output=True, text=True)
print('mambapy', chk.stdout.strip(), (ins.stderr or chk.stderr).strip()[:200], flush=True)
assert 'pscan OK' in chk.stdout, 'mambapy NO instalado'
cmd = [sys.executable, '-u', 'entrenar_real.py', '--condicion', '$COND', '--semilla', '$SEM',
       '--pasos', '$PASOS', '--batch', '8', '--acum', '1', '--largo', '192',
       '--n-hechos', '$NH',
       '--modelo', '$MODELO', '--cada', '100',
       '--salida', '/content/${UNI}.json']
log = open('/content/real.log', 'w')
p = subprocess.Popen(cmd, cwd='/content/real', stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
open('/content/real.pid', 'w').write(str(p.pid))
print('lanzado pid', p.pid, flush=True)
PY
timeout -k 30 900 "${CL[@]}" exec -s "$SESION" --timeout 600 -f "$TMP/lanzar.py" || exit 1

cat > "$TMP/ver.py" <<'PY'
import os
try:
    pid = int(open('/content/real.pid').read())
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
    print('ULTIMO=', [l for l in open('/content/real.log') if l.strip()][-1].strip()[:200])
except Exception:
    print('ULTIMO= (sin log)')
PY

for i in $(seq 1 60); do          # hasta 2 h
  sleep 120
  OUT="$(timeout -k 20 180 "${CL[@]}" exec -s "$SESION" --timeout 120 -f "$TMP/ver.py" 2>&1 | tail -2)"
  echo "$OUT"
  echo "$OUT" | grep -q "VIVO= False" && break
done

timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/${UNI}.json" "$AQUI/${UNI}.json" 2>&1 | tail -1
timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/real.log" "$AQUI/${UNI}.log" 2>&1 | tail -1
timeout -k 30 120 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1
if [ -f "$AQUI/${UNI}.json" ]; then
  echo "== $UNI LISTO"
  mandar "🧪 modelo real · $UNI cerró
$(tail -4 "$AQUI/${UNI}.log" 2>/dev/null)"
else
  echo "== $UNI SIN RESULTADO"
fi
