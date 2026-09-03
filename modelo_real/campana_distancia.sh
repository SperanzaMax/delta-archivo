#!/usr/bin/env bash
# Corre VARIAS unidades del experimento de distancia en UNA SOLA sesion de Colab · 3-sep
#
# A diferencia de `correr_real_colab.sh`, que gasta una asignacion por unidad, esto lanza un unico
# proceso que recorre la lista entera. Con el scan paralelo cada unidad de 400 pasos cuesta minutos,
# asi que el overhead de pedir sesion pasaba a dominar el costo.
#
#   Uso:  campana_distancia.sh <CUENTA> <PASOS> <ETIQUETA> <cond:sem> [cond:sem ...]
#   Ej.:  campana_distancia.sh O 400 compuerta cerca:0 lejos:0
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
CUENTA="${1:?falta la cuenta}"; PASOS="${2:?faltan los pasos}"; ETIQ="${3:?falta la etiqueta}"
shift 3
TRABAJOS=("$@"); [ "${#TRABAJOS[@]}" -eq 0 ] && { echo "faltan trabajos cond:sem"; exit 2; }
NH="${NH:-4}"; LARGO="${LARGO:-64}"; BATCH="${BATCH:-8}"; CADA="${CADA:-100}"
MODELO="${MODELO:-state-spaces/mamba-130m-hf}"; N_EVAL="${N_EVAL:-32}"
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

VIVAS="$(timeout -k 20 180 "${CL[@]}" sessions 2>/dev/null | grep -iE "T4|L4|GPU" || true)"
SESION="$(echo "$VIVAS" | head -1 | sed -n 's/^\[\([^]]*\)\].*/\1/p')"
if [ -n "$SESION" ]; then
  echo "== $ETIQ en $CUENTA · REUSA $SESION"
else
  SESION="dist_${CUENTA,,}_$(date +%H%M)"
  echo "== $ETIQ en $CUENTA · sesion NUEVA $SESION"
  OUT="$(timeout -k 30 600 "${CL[@]}" new -s "$SESION" --gpu T4 2>&1 | tail -3)"; echo "$OUT"
  echo "$OUT" | grep -qi "READY" || { echo "sin sesion en $CUENTA"; exit 1; }
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; rm -f "$lk"' EXIT
tar czf "$TMP/real.tgz" tarea_real.py entrenar_real.py vocabulario.json campana_remota.py
timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$TMP/real.tgz" /content/real.tgz || exit 1

LISTA="$(printf '%s ' "${TRABAJOS[@]}")"
cat > "$TMP/lanzar.py" <<PY
import os, subprocess, sys
os.makedirs('/content/real', exist_ok=True)
subprocess.run('tar xzf /content/real.tgz -C /content/real', shell=True, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mambapy'], check=False)
chk = subprocess.run([sys.executable, '-c', 'from mambapy.pscan import pscan; print("pscan OK")'],
                     capture_output=True, text=True)
print('mambapy', chk.stdout.strip(), chk.stderr.strip()[:200], flush=True)
assert 'pscan OK' in chk.stdout, 'mambapy NO instalado'
# El recorrido vive en campana_remota.py, que se sube tal cual. Generarlo desde el heredoc costo dos
# arranques de sesion el 3-sep: la indentacion de dedent, y \\n volviendose salto de linea real.
env = dict(os.environ, TRABAJOS='$LISTA', PASOS='$PASOS', BATCH='$BATCH', LARGO='$LARGO',
           NH='$NH', CADA='$CADA', MODELO='$MODELO', ETIQ='$ETIQ', N_EVAL='$N_EVAL')
log = open('/content/campana_$ETIQ.log', 'w')
p = subprocess.Popen([sys.executable, '-u', '/content/real/campana_remota.py'],
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
open('/content/campana.pid', 'w').write(str(p.pid))
print('lanzado pid', p.pid, 'trabajos: $LISTA', flush=True)
PY
timeout -k 30 900 "${CL[@]}" exec -s "$SESION" --timeout 600 -f "$TMP/lanzar.py" || exit 1

cat > "$TMP/ver.py" <<PY
import os, glob
try:
    pid = int(open('/content/campana.pid').read())
    vivo = os.path.exists('/proc/%d' % pid)
    if vivo:
        try:
            if open('/proc/%d/stat' % pid).read().split(')')[-1].split()[0] == 'Z': vivo = False
        except Exception: pass
    print('VIVO=', vivo)
except Exception as e:
    print('VIVO= ?', e)
print('HECHOS=', sorted(os.path.basename(x) for x in glob.glob('/content/${ETIQ}_*.json')))
try:
    ls = [l for l in open('/content/campana_$ETIQ.log') if l.strip()]
    print('LOG:\n' + ''.join(ls[-10:]))
except Exception:
    print('(sin log)')
PY

for i in $(seq 1 90); do          # hasta 3 h
  sleep 120
  OUT="$(timeout -k 20 200 "${CL[@]}" exec -s "$SESION" --timeout 150 -f "$TMP/ver.py" 2>&1 | tail -16)"
  echo "--- $(date +%H:%M)"; echo "$OUT"
  # 3-sep: las sesiones de Colab se estan muriendo a los ~60 min y una semilla entera son ~80, asi
  # que bajar recien al final costaba la campania completa. Se baja CADA unidad apenas aparece.
  for t in "${TRABAJOS[@]}"; do
    U="${ETIQ}_${t%%:*}_s${t##*:}"
    [ -f "$AQUI/${U}.json" ] && continue
    echo "$OUT" | grep -q "${U}.json" || continue
    timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/${U}.json" "$AQUI/${U}.json" \
      2>&1 | tail -1
  done
  echo "$OUT" | grep -q "VIVO= False" && break
  if echo "$OUT" | grep -qi "not found"; then
    echo "!! la sesion $SESION se perdio; se sale para que el rotador cambie de cuenta"
    exit 3
  fi
done

for t in "${TRABAJOS[@]}"; do
  UNI="${ETIQ}_${t%%:*}_s${t##*:}"
  timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/${UNI}.json" "$AQUI/${UNI}.json" 2>&1 | tail -1
done
timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/campana_$ETIQ.log" "$AQUI/campana_$ETIQ.log" 2>&1 | tail -1
echo "== $ETIQ: sesion $SESION queda VIVA para reusar"
mandar "🧪 modelo real · $ETIQ cerró ($LISTA)
$(grep -hE 'condicion|BASELINE|eval '"$PASOS" "$AQUI/campana_$ETIQ.log" 2>/dev/null | tail -8)"
