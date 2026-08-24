#!/usr/bin/env bash
# Corre UN TRAMO de una corrida en una sesion de Colab que YA EXISTE, y se trae el checkpoint.
#
#   Uso:  tramo_colab.sh <CUENTA> <sesion> <nivel:semilla> <pasos_total> <tramo> [cada]
#   Ej.:  tramo_colab.sh F probe_F 4:0 20000 4000 1000
#
# La idea (2026-08-14, a pedido de Maxi): las VMs de Colab se caen sin aviso, asi que en vez de
# pelear por 75 minutos ininterrumpidos se corre de a tramos. El checkpoint vive EN LA PC —que es lo
# unico que no se cae— y viaja a la VM al empezar y de vuelta al terminar. El tramo siguiente puede
# correr en OTRA cuenta: el checkpoint no sabe ni le importa en que maquina se genero.
#
# Este script NO crea sesiones a proposito: la creacion de VMs es lo que gasta cuota, y quien decide
# cuando gastarla es Maxi. Recibe una sesion ya viva y la usa.
#
# Verificado antes de usarse (test local, 3 tramos de 20 pasos contra una corrida entera de 60):
# reanudar da el mismo resultado BIT A BIT que no haber cortado nunca.
set -uo pipefail

CUENTA="${1:?falta la cuenta}"
SESION="${2:?falta la sesion}"
UNIDAD="${3:?falta nivel:semilla}"
PASOS="${4:?faltan los pasos totales}"
TRAMO="${5:?falta el tramo}"
CADA="${6:-1000}"
# El horizonte de la lr se fija en el MAXIMO previsto (20000) aunque hoy se corte en 12000: asi
# «paramos en 12000 y si hace falta seguimos» no cambia la curva de aprendizaje a mitad de camino.
# Verificado: parar y extender da el mismo resultado bit a bit que correr de largo.
HORIZONTE="${HORIZONTE:-20000}"

NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
CKPTS="$AQUI/ckpts"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$SALIDA" "$CKPTS"

if [ "$CUENTA" = "A" ]; then
  CL=( "$COLAB" --auth adc )
else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
  CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
fi

# PREFIJO separa familias de corridas que comparten nivel y semilla pero NO son comparables:
# "n" = la campaña base (p_nose 0, todas las preguntas tienen respuesta en el archivo);
# "x" = la campaña de ABSTENCION (p_nose > 0, hay preguntas cuya respuesta no está).
# Sin esto las dos escribirían el mismo n4_s0.pkl y se pisarían el checkpoint.
PREFIJO="${PREFIJO:-n}"
P_NOSE="${P_NOSE:-0.0}"
# 2026-08-18 · campania de la CABEZA DE ABSTENCION (PREREG_CABEZA_ABSTENCION.md). Tres familias que
# comparten nivel y semilla y NO son comparables con la `x`, porque reinician Adam:
#   ABST=token  -> "t"   NOSE es una entrada mas del softmax (control pareado)
#   ABST=escala -> "s"   idem, renormalizando el vector de NOSE al entrar
#   ABST=cabeza -> "c"   salida binaria separada
ABST="${ABST:-token}"
REINIT="${REINIT:-1}"
# PREREG_FRONTERA.md: la campaña base de la frontera entrena DESDE CERO (SEMBRAR=0) y corta cuando
# `vigente` cruza los valores de CORTES, guardando un checkpoint por cruce.
SEMBRAR="${SEMBRAR:-1}"
CORTES="${CORTES:-}"
UNI="${PREFIJO}${NIVEL}_s${SEM}"

CK="$CKPTS/${UNI}.pkl"
# Siembra: la primera vez se arranca desde el checkpoint BASE ya saturado, igual que hizo la
# campania `x`. Se COPIA (no se mueve ni se comparte) para que la base quede intacta.
BASE="$CKPTS/n${NIVEL}_s${SEM}.pkl"
# PREREG_FRONTERA §3 · LAS 18 FASES. Una fase no arranca de la campaña base sino de un CORTE de la
# base de la frontera (`f2_s*.pkl.v85/90/95`), y cada corte cayó en un paso distinto porque el
# criterio fue el VALOR de `vigente`, no el paso. `fases.tsv` (lo escribe preparar_fases.py) trae
# para cada unidad su checkpoint de partida, su condicion y su paso final = corte + 2000. Los tres
# datos salen de ahi y no del entorno, para que no puedan desincronizarse entre scripts.
if [ -f "$AQUI/fases.tsv" ]; then
  FILA="$(awk -F'\t' -v u="$UNI" '$1==u {print; exit}' "$AQUI/fases.tsv")"
  if [ -n "$FILA" ]; then
    BASE="$CKPTS/$(echo "$FILA" | cut -f2)"
    PASOS="$(echo "$FILA" | cut -f4)"
    ABST="$(echo "$FILA" | cut -f5)"
    SEMBRAR=1; REINIT=1; CORTES=""
    echo "== fase de la frontera: $UNI parte de $(basename "$BASE") · abst $ABST · hasta $PASOS"
  fi
fi
SEMBRADO=0
if [ "$SEMBRAR" = "1" ] && [ ! -f "$CK" ] && [ -f "$BASE" ]; then
  echo "== siembra: $UNI arranca desde $(basename "$BASE")"
  cp "$BASE" "$CK"
  SEMBRADO=1
fi
JS="$SALIDA/${UNI}.json"
echo "== tramo · cuenta $CUENTA · sesion $SESION · $UNI · +$TRAMO de $PASOS pasos · p_nose $P_NOSE · abst $ABST"

tar czf "$TMP/micro.tgz" -C "$AQUI" idioma.py datos.py modelo.py entrenar.py chequeo_padding.py
timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$TMP/micro.tgz" /content/micro.tgz || exit 1
if [ -f "$CK" ]; then
  echo "== subiendo checkpoint previo ($(du -h "$CK" | cut -f1))"
  timeout -k 30 420 "${CL[@]}" upload -s "$SESION" "$CK" /content/ck.pkl || exit 1
fi
timeout -k 30 420 "${CL[@]}" install -s "$SESION" optax >/dev/null 2>&1

cat > "$TMP/lanzar.py" <<PY
import os, subprocess, sys
os.makedirs('/content/micro', exist_ok=True); os.makedirs('/content/salidas', exist_ok=True)
subprocess.run('tar xzf /content/micro.tgz -C /content/micro', shell=True, check=True)
# La deteccion del acelerador va en un SUBPROCESO que muere enseguida, no con un import acá
# (2026-08-15). En TPU, el proceso que hace el import de jax se queda con el chip TOMADO, y este
# proceso es el kernel de Colab: sobrevive todo el tramo. El entrenamiento arrancaba después y
# moría con "The TPU is already in use by process with pid N". Costó las 5 únicas TPU que Colab
# nos asignó en todo el día —el 13 % de las asignaciones conseguidas— y ninguna llegó a entrenar.
# En GPU nunca se notó porque CUDA admite varios procesos sobre el mismo dispositivo.
det = subprocess.run([sys.executable, '-c',
                      'import jax; print(jax.__version__); print(jax.devices())'],
                     capture_output=True, text=True)
print('jax', det.stdout.strip().replace('\n', ' | '), flush=True)
assert ('CudaDevice' in det.stdout or 'TpuDevice' in det.stdout), 'NO hay acelerador'
chk = subprocess.run([sys.executable, 'chequeo_padding.py'], cwd='/content/micro',
                     capture_output=True, text=True)
assert 'compuerta ABRE' in chk.stdout, 'la compuerta de padding NO abre'
print('compuerta de padding OK', flush=True)
cmd = [sys.executable, '-u', 'entrenar.py', '--nivel', '$NIVEL', '--semilla', '$SEM',
       '--pasos', '$PASOS', '--tramo', '$TRAMO', '--cada', '$CADA', '--d', '128', '--capas', '4',
       '--lr', '1e-3', '--p-vieja', '0.35', '--idioma', '2', '--horizonte', '$HORIZONTE',
       '--p-nose', '$P_NOSE', '--abst', '$ABST',
       *(['--cortes-vigente', '$CORTES'] if '$CORTES' else []),
       '--salida', '/content/salidas/${UNI}.json', '--ckpt', '/content/ck.pkl']
if '$REINIT' == '1' and '$SEMBRADO' == '1':
    # Adam se reinicia SOLO al entrar en la fase (primer tramo, sembrado desde la base), nunca al
    # continuar un tramo posterior: reiniciarlo a mitad haria que la corrida deje de ser una sola.
    cmd.append('--reinit-adam')
log = open('/content/micro.log', 'w')
p = subprocess.Popen(cmd, cwd='/content/micro', stdout=log, stderr=subprocess.STDOUT,
                     start_new_session=True)
open('/content/micro.pid', 'w').write(str(p.pid))
print('lanzado pid', p.pid, flush=True)
PY
timeout -k 30 300 "${CL[@]}" exec -s "$SESION" -f "$TMP/lanzar.py" || exit 1

cat > "$TMP/ver.py" <<'PY'
import json, os
try:
    pid = int(open('/content/micro.pid').read()); print('VIVO=', os.path.exists('/proc/%d' % pid))
except Exception as e:
    print('VIVO= ?', e)
try:
    ls = [l for l in open('/content/micro.log', errors='ignore') if l.strip()]
    print('ULTIMO=', ls[-1].strip())
except Exception:
    pass
for f in sorted(os.listdir('/content/salidas')):
    if f.endswith('.json'):
        print('@@JSON@@', f, json.dumps(json.load(open('/content/salidas/' + f))))
PY

# Presupuesto de polling. Medido en la T4 de hoy: ~0,46 s/paso (el 13-ago era 0,22 — la T4 que toca
# no siempre rinde igual), o sea ~8 min cada 1000 pasos. Se calcula con el doble de margen: si el
# polling corta antes que el entrenamiento, el tramo se pierde aunque la VM lo haya terminado.
MIN=$(( TRAMO / 1000 * 10 + 20 ))
echo "== polling cada 2 min (presupuesto ~${MIN} min)"
LISTO=0
TICK=0
for _ in $(seq 1 $(( MIN / 2 ))); do
  sleep 120
  TICK=$((TICK + 1))
  OUT="$(timeout -k 30 240 "${CL[@]}" exec -s "$SESION" -f "$TMP/ver.py" 2>&1 || true)"

  # El checkpoint se BAJA cada ~8 min, no sólo al final del tramo. El 14-ago la VM murio a mitad
  # del tramo 2 y se perdieron 2000 pasos de computo: el JSON habia llegado por streaming al paso
  # 6000, pero el ultimo checkpoint en la PC era el del 4000, porque sólo se bajaba al terminar.
  # Son 10 MB cada 8 min; a cambio, lo que se pierde cuando cae la VM baja de un tramo entero a un
  # solo intervalo de evaluacion.
  if [ $((TICK % 4)) -eq 0 ]; then
    if timeout -k 30 300 "${CL[@]}" download -s "$SESION" /content/ck.pkl "$TMP/ck_p.pkl" >/dev/null 2>&1 \
       && [ -s "$TMP/ck_p.pkl" ]; then
      mv "$TMP/ck_p.pkl" "$CK"
      echo "   [checkpoint parcial guardado: paso $(grep -o '"paso": [0-9]*' "$JS" 2>/dev/null | tail -1 | grep -o '[0-9]*')]"
    fi
    for v in $(echo "$CORTES" | tr ',' ' '); do
      vv=$(awk -v x="$v" 'BEGIN{printf "%d", x*100+0.5}')   # awk, no printf: printf usa el locale
      if [ ! -f "$CK.v$vv" ] \
         && timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/ck.pkl.v$vv" "$TMP/c.pkl" >/dev/null 2>&1 \
         && [ -s "$TMP/c.pkl" ]; then
        mv "$TMP/c.pkl" "$CK.v$vv"; echo "   [corte vigente $v bajado -> $(basename "$CK.v$vv")]"
      fi
    done
  fi
  { printf '%s\n' "$OUT" | grep '^@@JSON@@ ' || true; } | while read -r _ nombre resto; do
    printf '%s' "$resto" > "$SALIDA/$nombre"
  done
  { printf '%s\n' "$OUT" | grep -E "VIVO=|ULTIMO=" || true; } | tr '\n' ' '; echo
  if printf '%s' "$OUT" | grep -qE "not found|appears to be lost"; then
    echo "!! sesion perdida — el checkpoint de la PC conserva el ultimo tramo bajado"; break
  fi
  if printf '%s' "$OUT" | grep -q "VIVO= False"; then LISTO=1; echo "== tramo terminado"; break; fi
  # El PID no alcanza: un proceso ZOMBIE conserva /proc/<pid>, asi que VIVO se queda en True para
  # siempre y el tramo muere por agotamiento del polling sin bajar nada (paso el 18-ago con las tres
  # bases de la frontera). El hecho de que el tramo termino es que el JSON llego al paso pedido.
  if [ -f "$JS" ] && grep -q "\"paso\": $PASOS" "$JS" 2>/dev/null; then
    LISTO=1; echo "== tramo terminado (JSON en el paso $PASOS)"; break
  fi
done

if [ "$LISTO" = "1" ]; then
  echo "== bajando checkpoint"
  timeout -k 30 420 "${CL[@]}" download -s "$SESION" /content/ck.pkl "$TMP/ck.pkl" >/dev/null 2>&1 \
    && mv "$TMP/ck.pkl" "$CK" && echo "   checkpoint en $CK ($(du -h "$CK" | cut -f1))"
  # Los cortes por valor de `vigente` son el PRODUCTO de la campaña base: si no se bajan, el tramo
  # no sirvio de nada. Se intenta cada uno por separado; los que no existan fallan en silencio.
  for v in $(echo "$CORTES" | tr ',' ' '); do
    vv=$(awk -v x="$v" 'BEGIN{printf "%d", x*100+0.5}')
    if timeout -k 30 420 "${CL[@]}" download -s "$SESION" "/content/ck.pkl.v$vv" "$TMP/c.pkl" >/dev/null 2>&1 \
       && [ -s "$TMP/c.pkl" ]; then
      mv "$TMP/c.pkl" "$CK.v$vv"; echo "   corte vigente $v -> $(basename "$CK.v$vv")"
    fi
  done
  if [ -f "$JS" ]; then
    echo "   ultimo paso registrado: $(grep -o '"paso": [0-9]*' "$JS" | tail -1)"
  fi
fi
echo "== fin del tramo (la sesion queda VIVA para el tramo siguiente; pararla con: colab stop -s $SESION)"
