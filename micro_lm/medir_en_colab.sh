#!/usr/bin/env bash
# Corre un script de MEDICION en una VM de Colab, en vez de en la PC.
#
# Nace de la regla que puso Maxi el 31-ago: «prioriza para correr en esta PC cosas que no lleven
# mucho tiempo constante de CPU; para eso tenemos Colab». Ese dia una sonda estuvo 61 minutos al 93 %
# en la maquina que el usa. `tramo_abst.sh` ya sabia mandar el ENTRENAMIENTO a Colab; esto es su
# hermano para las MEDICIONES, que es lo que faltaba.
#
#   Uso:  medir_en_colab.sh <CUENTA> <script.py> <ckpt1> [ckpt2 ...]
#   Ej.:  medir_en_colab.sh A sonda_techo_curva.py ckpts/n3_s0.pkl ckpts/t03_s3.pkl
#
# Lo que hace, y en este orden:
#   1. pide una T4 (con TPU de respaldo, igual que el rotador),
#   2. sube el codigo y los checkpoints que se le pasen —a `ckpts/` dentro de la VM, con el MISMO
#      nombre, para que el script los encuentre por la ruta relativa de siempre—,
#   3. corre el script con `python -u` (la otra leccion del 31: sin `-u` una corrida estuvo una hora
#      sin imprimir una linea y no se supo que estaba trabada),
#   4. baja los .json que el script haya dejado,
#   5. PARA la sesion pase lo que pase, tambien si el script revienta.
#
# ✅ PROBADO el 1-sep 07:18 con `smoke_medicion.py` (cuenta C, T4): subio codigo + 2 checkpoints,
# corrio, y el .json volvio a la PC. jax 0.11.1 sobre CudaDevice. La cuenta A dio 503, o sea que
# conviene rotar cuentas igual que hace el rotador de entrenamiento.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"
COLAB=/home/maxi/.venv-colab-cli/bin/colab

CUENTA="${1:?falta la cuenta (A, C..N)}"
SCRIPT="${2:?falta el script de medicion}"
shift 2
CKPTS=("$@")
[ "${#CKPTS[@]}" -eq 0 ] && { echo "falta al menos un checkpoint"; exit 1; }

if [ "$CUENTA" = "A" ]; then
  CL=( "$COLAB" --auth adc )
else
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"
  CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" )
fi
SESION="med_${CUENTA,,}_$(date +%H%M)"
TMP="$(mktemp -d)"
trap 'timeout -k 20 120 "${CL[@]}" stop -s "$SESION" >/dev/null 2>&1; rm -rf "$TMP"' EXIT

echo "== pidiendo acelerador en la cuenta $CUENTA (sesion $SESION)"
if timeout -k 30 420 "${CL[@]}" new -s "$SESION" --gpu T4 >/dev/null 2>&1; then ACC=T4
elif timeout -k 30 420 "${CL[@]}" new -s "$SESION" --tpu v5e1 >/dev/null 2>&1; then ACC="TPU v5e1"
else echo "   503: ni T4 ni TPU en $CUENTA. Probar otra cuenta."; exit 1; fi
echo "   conseguido: $ACC"

echo "== subiendo codigo"
tar czf "$TMP/micro.tgz" -C "$AQUI" \
  idioma.py datos.py modelo.py entrenar.py medir_ratio_ce.py sonda_volado.py "$SCRIPT" \
  $(ls sonda_techo.py 2>/dev/null)
timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$TMP/micro.tgz" /content/micro.tgz || exit 1

cat > "$TMP/prep.py" <<'PYPREP'
import os, subprocess
os.makedirs('/content/micro/ckpts', exist_ok=True)
subprocess.run('tar xzf /content/micro.tgz -C /content/micro', shell=True, check=True)
print('codigo descomprimido')
PYPREP
timeout -k 30 180 "${CL[@]}" exec -s "$SESION" --timeout 120 -f "$TMP/prep.py" 2>&1 | tail -1
timeout -k 30 420 "${CL[@]}" install -s "$SESION" optax >/dev/null 2>&1

for CK in "${CKPTS[@]}"; do
  echo "== subiendo $(basename "$CK") ($(du -h "$CK" | cut -f1))"
  timeout -k 30 420 "${CL[@]}" upload -s "$SESION" "$AQUI/$CK" "/content/micro/ckpts/$(basename "$CK")" || exit 1
done

# Los checkpoints subidos se le pasan al script COMO ARGUMENTOS: asi mide exactamente los que se
# mandaron, y no depende de una lista hardcodeada adentro del script (que fue lo que obligo a tocar
# `sonda_techo_curva.py` el 1-sep para medir 17 unidades en vez de sus 10 por defecto).
ARGS=""
for CK in "${CKPTS[@]}"; do ARGS="$ARGS'ckpts/$(basename "$CK")',"; done
cat > "$TMP/correr.py" <<PY
import subprocess, sys, glob
# -u por la leccion del 31: sin esto la salida queda en el buffer y no se sabe si avanza o se trabo.
p = subprocess.run([sys.executable, '-u', '$SCRIPT'] + [$ARGS], cwd='/content/micro',
                   capture_output=True, text=True)
print(p.stdout[-8000:])
if p.returncode: print('STDERR:', p.stderr[-3000:])
print('JSON generados:', glob.glob('/content/micro/*.json'))
PY
echo "== corriendo $SCRIPT en la VM"
timeout -k 60 3000 "${CL[@]}" exec -s "$SESION" --timeout 2700 -f "$TMP/correr.py" 2>&1 | tail -60

cat > "$TMP/listar.py" <<'PYL'
import glob
print('\n'.join(glob.glob('/content/micro/*.json')))
PYL
for J in $(timeout -k 30 180 "${CL[@]}" exec -s "$SESION" --timeout 120 -f "$TMP/listar.py" 2>/dev/null | grep '^/content'); do
  echo "== bajando $(basename "$J")"
  timeout -k 30 300 "${CL[@]}" download -s "$SESION" "$J" "$AQUI/$(basename "$J")" >/dev/null 2>&1 \
    && echo "   -> $(basename "$J")" || echo "   ** no se pudo bajar $J **"
done
echo "== listo (la sesion se para sola al salir)"
