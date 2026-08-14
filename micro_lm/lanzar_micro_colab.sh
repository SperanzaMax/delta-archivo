#!/usr/bin/env bash
# Lanza corridas del MICRO-LM en una VM de Colab T4, via el CLI oficial (google-colab-cli).
#
#   Uso:  lanzar_micro_colab.sh <CUENTA A-J> <unidades> [pasos]
#   Ej.:  lanzar_micro_colab.sh A 1:0,2:1        20000
#
# Una «unidad» es `nivel:semilla`. Se corren en SECUENCIA dentro de la misma VM.
#
# ## Reparto entre cuentas: por semilla, nunca por nivel
# El contraste que se quiere medir es entre NIVELES. Si cada nivel viviera en una sola cuenta,
# cualquier diferencia entre cuentas (tipo de GPU, version de libreria) caeria justo adentro del
# contraste. Por eso `campania_micro.sh` esparce cada nivel entre varias cuentas. Y todas piden T4:
# el script ABORTA si le dan otra cosa.
#
# ## Por que NO se usa un `colab exec` largo (leccion del 2026-08-09, costo dos VMs)
# `colab exec` espera output del kernel y aborta con «TimeoutError» si el script tarda en escribir.
# El patron correcto es lanzar en background dentro de la VM (Popen + log a archivo) y hacer polling
# con execs CORTOS que leen el log por offset. Ningun exec dura mas de unos segundos.
#
# ## Rescate por streaming
# `entrenar.py` reescribe su JSON en cada evaluacion (cada 2000 pasos). El polling se los trae en
# cada tick, asi que si la VM se muere a mitad de camino igual queda en disco todo lo evaluado hasta
# ahi. Los pesos, que son 3,5 MB, se bajan recien al final.
set -euo pipefail

CUENTA="${1:?falta la cuenta: A-J}"
UNIDADES="${2:?faltan las unidades, p.ej. 1:0,2:1}"
PASOS="${3:-20000}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
SESION="micro_${CUENTA,,}${SUF:-}"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Perfiles: A es el ADC por defecto; el resto necesita SU CLOUDSDK_CONFIG (identidad) **y** su
# --config (estado de sesiones). Sin lo primero comparten identidad; sin lo segundo se pisan el
# sessions.json y la sesion queda inalcanzable por CLI.
case "$CUENTA" in
  A) CL=( "$COLAB" --auth adc ) ;;
  B|C|D|E|F|G|H|I|J)
     PERFIL="$HOME/.gcloud-cuenta$CUENTA"
     if [ ! -f "$PERFIL/application_default_credentials.json" ]; then
       echo "La cuenta $CUENTA no esta dada de alta. Corre UNA VEZ (es interactivo):"
       echo "  CLOUDSDK_CONFIG=$PERFIL gcloud auth application-default login \\"
       echo "    --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\\"
       echo "https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory"
       exit 3
     fi
     export CLOUDSDK_CONFIG="$PERFIL"
     CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" ) ;;
  *) echo "cuenta invalida: $CUENTA (usar A-J)"; exit 2 ;;
esac

echo "== cuenta $CUENTA · unidades $UNIDADES · $PASOS pasos"
mkdir -p "$SALIDA"
tar czf "$TMP/micro.tgz" -C "$AQUI" idioma.py datos.py modelo.py entrenar.py chequeo_padding.py

if ! "${CL[@]}" sessions 2>/dev/null | grep -q "\[$SESION\]"; then
  echo "== creando sesion T4"
  "${CL[@]}" new -s "$SESION" --gpu T4
else
  echo "== reusando sesion viva $SESION"
fi

"${CL[@]}" upload -s "$SESION" "$TMP/micro.tgz" /content/micro.tgz
"${CL[@]}" install -s "$SESION" optax

# --- prep + lanzamiento en background: un solo exec CORTO ---
cat > "$TMP/lanzar.py" <<PY
import os, subprocess, sys
os.makedirs('/content/micro', exist_ok=True)
os.makedirs('/content/salidas', exist_ok=True)
subprocess.run('tar xzf /content/micro.tgz -C /content/micro', shell=True, check=True)

import jax
devs = jax.devices()
print('jax', jax.__version__, '| devices', devs, flush=True)
assert any(d.platform == 'gpu' for d in devs), 'NO hay GPU: la campania exige T4 homogenea'

# La compuerta de padding, ANTES de gastar una hora de GPU. Es lo que fallo el 13-ago.
chk = subprocess.run([sys.executable, 'chequeo_padding.py'], cwd='/content/micro',
                     capture_output=True, text=True)
print(chk.stdout, flush=True)
assert 'compuerta ABRE' in chk.stdout, 'la compuerta de padding NO abre: no se corre nada'

guion = '''
set -e
cd /content/micro
for u in \$(echo "$UNIDADES" | tr ',' ' '); do
  n=\${u%%:*}; s=\${u##*:}
  echo "@@INICIO@@ nivel \$n semilla \$s"
  python -u entrenar.py --nivel \$n --semilla \$s --pasos $PASOS --d 128 --capas 4 \\
      --lr 1e-3 --p-vieja 0.35 \\
      --salida /content/salidas/n\${n}_s\${s}.json \\
      --pesos /content/salidas/n\${n}_s\${s}.pkl
  echo "@@FIN@@ nivel \$n semilla \$s"
done
echo "@@TODO_LISTO@@"
'''
open('/content/correr.sh', 'w').write(guion)
log = open('/content/micro.log', 'w')
p = subprocess.Popen(['bash', '/content/correr.sh'], stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
open('/content/micro.pid', 'w').write(str(p.pid))
print('runner lanzado, pid', p.pid, flush=True)
PY
"${CL[@]}" exec -s "$SESION" -f "$TMP/lanzar.py"

# --- polling: execs cortos, log por offset + JSON parciales en cada tick ---
cat > "$TMP/ver.py" <<'PY'
import json, os
OFF, LOG = '/content/micro.offset', '/content/micro.log'
try:
    pid = int(open('/content/micro.pid').read()); print('VIVO=', os.path.exists(f'/proc/{pid}'))
except Exception as e:
    print('VIVO= ?', e)
off = 0
if os.path.exists(OFF):
    try: off = int(open(OFF).read().strip())
    except ValueError: off = 0
if os.path.exists(LOG):
    if os.path.getsize(LOG) < off: off = 0
    with open(LOG, errors='ignore') as f:
        f.seek(off); nuevo = f.read()
    open(OFF, 'w').write(str(off + len(nuevo)))
    print(nuevo, end='')
# los JSON parciales viajan enteros en cada tick: si la VM muere, no se pierde lo ya evaluado
for f in sorted(os.listdir('/content/salidas')):
    if f.endswith('.json'):
        print('@@JSON@@', f, json.dumps(json.load(open('/content/salidas/' + f))))
PY

MIN=$(( (PASOS / 1000 * 5 + 20) * $(echo "$UNIDADES" | tr ',' ' ' | wc -w) ))
echo "== corriendo · polling cada 2 min · presupuesto ~${MIN} min"
for _ in $(seq 1 $(( MIN / 2 ))); do
  sleep 120
  OUT="$("${CL[@]}" exec -s "$SESION" -f "$TMP/ver.py" 2>&1 || true)"
  # `|| true` en los dos: un grep sin coincidencias devuelve 1 y, con `set -e` + `pipefail`, mataria
  # el lanzador justo en el primer tick, cuando todavia no hay ningun JSON.
  { printf '%s\n' "$OUT" | grep '^@@JSON@@ ' || true; } | while read -r _ nombre resto; do
    printf '%s' "$resto" > "$SALIDA/$nombre"
  done
  { printf '%s\n' "$OUT" | grep -vE '^@@JSON@@ ' \
      | grep -E "VIVO=|eval:|@@|Error|Traceback|ABORTA" || true; } | tail -6
  if printf '%s' "$OUT" | grep -q "VIVO= False"; then echo "== runner terminado"; break; fi
done

echo "== bajando pesos y JSON finales"
cat > "$TMP/pack.py" <<'PY'
import subprocess
subprocess.run('cd /content/salidas && tar czf /content/micro_out.tgz .', shell=True)
print('empaquetado')
PY
"${CL[@]}" exec -s "$SESION" -f "$TMP/pack.py"
"${CL[@]}" download -s "$SESION" /content/micro_out.tgz "$TMP/micro_out.tgz" || echo "!! no se pudo bajar"
[ -f "$TMP/micro_out.tgz" ] && tar xzf "$TMP/micro_out.tgz" -C "$SALIDA"
echo "en $SALIDA: $(ls "$SALIDA"/n*_s*.json 2>/dev/null | wc -l) JSON · $(ls "$SALIDA"/n*_s*.pkl 2>/dev/null | wc -l) pesos"

echo "== parando la VM (no dejar unidades quemandose)"
"${CL[@]}" stop -s "$SESION" || true
