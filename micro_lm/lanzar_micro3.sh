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
#
# ## REINTENTOS (2026-08-14, despues de perder 8 VMs de una)
# La campania de la mañana se cayo entera en una ventana de una hora: siete VMs murieron del lado
# del servidor con tres sintomas distintos («Connection was lost», «Timeout waiting for output»,
# «Session not found») y el CLI se quedo polleando una sesion que ya no existia. Ahora:
#   · cada intento se rehace hasta MAX_INTENTOS veces, recreando la sesion;
#   · antes de cada intento se descartan las unidades YA COMPLETAS (su JSON local llego a `pasos`),
#     asi un reintento no vuelve a correr lo que ya se logro;
#   · el polling detecta que la sesion se perdio y corta en vez de seguir hablandole al vacio;
#   · `new` reintenta ante 503/Service Unavailable, que es como Colab dice «ahora no hay T4».
set -uo pipefail

CUENTA="${1:?falta la cuenta: A-J}"
UNIDADES="${2:?faltan las unidades, p.ej. 1:0,2:1}"
PASOS="${3:-20000}"
MAX_INTENTOS="${MAX_INTENTOS:-3}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
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

mkdir -p "$SALIDA"
tar czf "$TMP/micro.tgz" -C "$AQUI" idioma.py datos.py modelo.py entrenar.py chequeo_padding.py

# Unidades que todavia faltan: las que no tienen JSON local con la ultima evaluacion en `PASOS`.
pendientes() {
  local out=()
  for u in $(echo "$UNIDADES" | tr ',' ' '); do
    local n="${u%%:*}" s="${u##*:}"
    local f="$SALIDA/n${n}_s${s}.json"
    if [ -f "$f" ] && grep -q "\"paso\": $PASOS" "$f" 2>/dev/null; then
      continue
    fi
    out+=("$u")
  done
  (IFS=,; echo "${out[*]}")
}

# CASCADA DE ACELERADORES (2026-08-14, a pedido de Maxi: «si no te dan T4, fijate si te dan otra
# GPU»). Cuando Colab raciona las T4, una cuenta puede seguir teniendo otro acelerador libre.
# El orden va de menor a mayor: T4 es la de referencia y la que mas quota tiene; L4 y A100 se
# aceptan si estan disponibles. CPU queda AFUERA a proposito — medido en esta misma maquina, el
# entrenamiento en CPU va a ~1,9 s/paso, o sea 10,5 h para las 20000 unidades: no es una alternativa
# mas lenta, es no terminar nunca.
# El acelerador que toque queda registrado en el JSON (campo `hw` de `entrenar.py`), asi que si una
# celda queda rara se puede chequear si corrio en otro hardware antes de buscarle una explicacion.
ACELERADORES="${ACELERADORES:-T4 L4 A100}"

crear_sesion() {
  local s="$1"
  for i in 1 2; do
    for acc in $ACELERADORES; do
      if timeout 420 "${CL[@]}" new -s "$s" --gpu "$acc" 2>&1 | tee "$TMP/new.log" | tail -2; then
        if grep -q "READY" "$TMP/new.log"; then
          echo "   >> asigno $acc en la cuenta $CUENTA"
          return 0
        fi
      fi
      # 503 = sin ese acelerador ahora; «rejected/quota» = la cuenta no tiene derecho a ese.
      # Los dos casos se tratan igual: probar el siguiente de la lista.
      echo "   (sin $acc)"
    done
    echo "   (ningun acelerador en la vuelta $i; espera 60 s)"
    sleep 60
  done
  return 1
}

# --- un intento completo: crear, subir, lanzar, pollear -----------------------------------------
intento() {
  local n_intento="$1" faltan="$2"
  local SESION="micro_${CUENTA,,}_${n_intento}"
  echo "== cuenta $CUENTA · intento $n_intento · unidades $faltan · $PASOS pasos"

  crear_sesion "$SESION" || { echo "!! no se pudo asignar T4"; return 1; }
  timeout 300 "${CL[@]}" upload -s "$SESION" "$TMP/micro.tgz" /content/micro.tgz || return 1
  timeout 420 "${CL[@]}" install -s "$SESION" optax || return 1

  cat > "$TMP/lanzar.py" <<PY
import os, subprocess, sys
os.makedirs('/content/micro', exist_ok=True)
os.makedirs('/content/salidas', exist_ok=True)
subprocess.run('tar xzf /content/micro.tgz -C /content/micro', shell=True, check=True)

import jax
devs = jax.devices()
print('jax', jax.__version__, '| devices', devs, flush=True)
assert any(d.platform == 'gpu' for d in devs), 'NO hay GPU: en CPU son 10,5 h por unidad'

# La compuerta de padding, ANTES de gastar una hora de GPU. Es lo que fallo el 13-ago.
chk = subprocess.run([sys.executable, 'chequeo_padding.py'], cwd='/content/micro',
                     capture_output=True, text=True)
print(chk.stdout, flush=True)
assert 'compuerta ABRE' in chk.stdout, 'la compuerta de padding NO abre: no se corre nada'

guion = '''
set -e
cd /content/micro
for u in \$(echo "$faltan" | tr ',' ' '); do
  n=\${u%%:*}; s=\${u##*:}
  echo "@@INICIO@@ nivel \$n semilla \$s"
  python -u entrenar.py --nivel \$n --semilla \$s --pasos $PASOS --d 128 --capas 4 \\
      --lr 1e-3 --p-vieja 0.35 --idioma ${IDIOMA:-2} \\
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
  timeout 300 "${CL[@]}" exec -s "$SESION" -f "$TMP/lanzar.py" || return 1

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

  local MIN=$(( (PASOS / 1000 * 5 + 25) * $(echo "$faltan" | tr ',' ' ' | wc -w) ))
  echo "== corriendo · polling cada 2 min · presupuesto ~${MIN} min"
  local perdidas=0 term=1
  for _ in $(seq 1 $(( MIN / 2 ))); do
    sleep 120
    OUT="$(timeout 240 "${CL[@]}" exec -s "$SESION" -f "$TMP/ver.py" 2>&1 || true)"
    { printf '%s\n' "$OUT" | grep '^@@JSON@@ ' || true; } | while read -r _ nombre resto; do
      printf '%s' "$resto" > "$SALIDA/$nombre"
    done
    { printf '%s\n' "$OUT" | grep -vE '^@@JSON@@ ' \
        | grep -E "VIVO=|eval:|@@|Error|Traceback|ABORTA" || true; } | tail -4

    # La sesion se murio del lado del servidor: cortar y dejar que el intento siguiente la recree.
    if printf '%s' "$OUT" | grep -qE "not found|appears to be lost|Connection was lost|401|404"; then
      perdidas=$((perdidas + 1))
      if [ "$perdidas" -ge 2 ]; then
        echo "!! sesion perdida del lado del servidor"; term=0; break
      fi
    else
      perdidas=0
    fi
    if printf '%s' "$OUT" | grep -q "@@TODO_LISTO@@"; then echo "== unidades terminadas"; break; fi
    if printf '%s' "$OUT" | grep -q "VIVO= False"; then echo "== runner terminado"; break; fi
  done

  if [ "$term" = "1" ]; then
    echo "== bajando pesos y JSON finales"
    cat > "$TMP/pack.py" <<'PY'
import subprocess
subprocess.run('cd /content/salidas && tar czf /content/micro_out.tgz .', shell=True)
print('empaquetado')
PY
    timeout 300 "${CL[@]}" exec -s "$SESION" -f "$TMP/pack.py" >/dev/null 2>&1 \
      && timeout 420 "${CL[@]}" download -s "$SESION" /content/micro_out.tgz "$TMP/micro_out.tgz" >/dev/null 2>&1 \
      && tar xzf "$TMP/micro_out.tgz" -C "$SALIDA" && echo "   bajado OK"
  fi

  timeout 180 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1 || true
  [ "$term" = "1" ]
}

for k in $(seq 1 "$MAX_INTENTOS"); do
  FALTAN="$(pendientes)"
  if [ -z "$FALTAN" ]; then echo "== cuenta $CUENTA: todas las unidades completas"; break; fi
  if intento "$k" "$FALTAN"; then
    [ -z "$(pendientes)" ] && { echo "== cuenta $CUENTA lista"; break; }
  fi
  echo "== cuenta $CUENTA: quedan pendientes ($(pendientes)), reintento en 60 s"
  sleep 60
done

echo "== cuenta $CUENTA fin · $(ls "$SALIDA"/n*_s*.json 2>/dev/null | wc -l) JSON en $SALIDA"
