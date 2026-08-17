#!/usr/bin/env bash
# Corre UN TRAMO en una GPU alquilada por hora (RunPod, Vast, Lambda, una VM propia: cualquier cosa
# que se alcance por SSH), y se trae el checkpoint.
#
#   Uso:  tramo_ssh.sh <destino_ssh> <nivel:semilla> <pasos_total> <tramo> [cada]
#   Ej.:  tramo_ssh.sh root@1.2.3.4:22 4:2 12000 8000 500
#         PUERTO va pegado con ':' porque los proveedores spot casi nunca dan el 22.
#
# Gemelo de tramo_colab.sh (2026-08-17). Misma idea y mismas garantias: el checkpoint vive EN LA PC
# —lo unico que no se cae— y viaja a la maquina al empezar y de vuelta al terminar; se baja cada
# ~8 min y no solo al final, porque una instancia spot desaparece sin aviso igual que una VM de
# Colab; y todo va con `timeout` porque colgarse para siempre fue el modo de falla mas caro del
# 14-ago. Reanudar da el mismo resultado bit a bit que no haber cortado nunca (verificado el 14-ago
# con 3 tramos de 20 pasos contra una corrida entera de 60).
#
# Lo que cambia respecto de Colab: el transporte es scp/ssh en vez de `colab upload/exec/download`,
# y no hay estado de sesion que se pueda pisar, asi que desaparece la regla del proceso unico.
set -uo pipefail

DEST="${1:?falta el destino ssh (user@host[:puerto])}"
UNIDAD="${2:?falta nivel:semilla}"
PASOS="${3:?faltan los pasos totales}"
TRAMO="${4:?falta el tramo}"
CADA="${5:-500}"
HORIZONTE="${HORIZONTE:-20000}"      # igual que en Colab: la lr se fija en el maximo previsto

HOST="${DEST%%:*}"
PUERTO="${DEST##*:}"; [ "$PUERTO" = "$DEST" ] && PUERTO=22
SSH=( ssh -p "$PUERTO" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "$HOST" )
SCP=( scp -P "$PUERTO" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 )

NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
CKPTS="$AQUI/ckpts"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$SALIDA" "$CKPTS"

PREFIJO="${PREFIJO:-n}"              # "n" = campaña base · "x" = campaña de abstencion (p_nose>0)
P_NOSE="${P_NOSE:-0.0}"
UNI="${PREFIJO}${NIVEL}_s${SEM}"
CK="$CKPTS/${UNI}.pkl"
JS="$SALIDA/${UNI}.json"
REMOTO="/root/micro"

echo "== tramo ssh · $HOST:$PUERTO · $UNI · +$TRAMO de $PASOS pasos · p_nose $P_NOSE"

tar czf "$TMP/micro.tgz" -C "$AQUI" idioma.py datos.py modelo.py entrenar.py chequeo_padding.py
timeout 120 "${SSH[@]}" "mkdir -p $REMOTO/salidas" || exit 1
timeout 300 "${SCP[@]}" "$TMP/micro.tgz" "$HOST:$REMOTO/micro.tgz" || exit 1
if [ -f "$CK" ]; then
  echo "== subiendo checkpoint previo ($(du -h "$CK" | cut -f1))"
  timeout 600 "${SCP[@]}" "$CK" "$HOST:$REMOTO/ck.pkl" || exit 1
fi

# Compuerta de padding y acelerador: las MISMAS dos guardas que en Colab, y por el mismo motivo.
# Si el padding trunca, la corrida mide el padding y no la tarea — fue lo que invalido los niveles
# 1-3 del 13-ago.
timeout 900 "${SSH[@]}" "cd $REMOTO && tar xzf micro.tgz && \
  pip -q install optax >/dev/null 2>&1; \
  python -c 'import jax; print(jax.devices())'" || exit 1
DEV="$(timeout 300 "${SSH[@]}" "cd $REMOTO && python -c 'import jax; print(jax.devices())'" 2>&1)"
echo "== dispositivos: $DEV"
case "$DEV" in *Cuda*|*Tpu*|*Rocm*) ;; *) echo "!! NO hay acelerador — se aborta"; exit 1;; esac
CHK="$(timeout 600 "${SSH[@]}" "cd $REMOTO && python chequeo_padding.py" 2>&1 | tail -3)"
case "$CHK" in *"compuerta ABRE"*) echo "== compuerta de padding OK";; *) echo "!! la compuerta de padding NO abre:"; echo "$CHK"; exit 1;; esac

# Se lanza DESPRENDIDO (setsid + nohup) y se pollea, en vez de esperar con la conexion abierta: si
# se corta el SSH, el entrenamiento sigue y el tramo no se pierde.
timeout 120 "${SSH[@]}" "cd $REMOTO && setsid nohup python -u entrenar.py \
  --nivel $NIVEL --semilla $SEM --pasos $PASOS --tramo $TRAMO --cada $CADA \
  --d 128 --capas 4 --lr 1e-3 --p-vieja 0.35 --idioma 2 --horizonte $HORIZONTE \
  --p-nose $P_NOSE --salida $REMOTO/salidas/${UNI}.json --ckpt $REMOTO/ck.pkl \
  > $REMOTO/micro.log 2>&1 < /dev/null & echo \$! > $REMOTO/micro.pid" || exit 1
echo "== lanzado pid $(timeout 60 "${SSH[@]}" "cat $REMOTO/micro.pid" 2>/dev/null)"

MIN=$(( TRAMO / 1000 * 10 + 20 ))
echo "== polling cada 2 min (presupuesto ~${MIN} min)"
LISTO=0; TICK=0
for _ in $(seq 1 $(( MIN / 2 ))); do
  sleep 120
  TICK=$((TICK + 1))
  OUT="$(timeout 180 "${SSH[@]}" "cd $REMOTO && \
    { [ -d /proc/\$(cat micro.pid) ] && echo 'VIVO= True' || echo 'VIVO= False'; }; \
    echo \"ULTIMO= \$(tail -1 micro.log 2>/dev/null)\"" 2>&1 || true)"

  if [ $((TICK % 4)) -eq 0 ]; then
    if timeout 420 "${SCP[@]}" "$HOST:$REMOTO/ck.pkl" "$TMP/ck_p.pkl" >/dev/null 2>&1 \
       && [ -s "$TMP/ck_p.pkl" ]; then
      mv "$TMP/ck_p.pkl" "$CK"
      timeout 300 "${SCP[@]}" "$HOST:$REMOTO/salidas/${UNI}.json" "$JS" >/dev/null 2>&1
      echo "   [checkpoint parcial guardado: paso $(grep -o '\"paso\": [0-9]*' "$JS" 2>/dev/null | tail -1 | grep -o '[0-9]*')]"
    fi
  fi
  printf '%s ' "$OUT" | tr '\n' ' '; echo
  if printf '%s' "$OUT" | grep -qE "Connection refused|No route to host|Connection timed out"; then
    echo "!! se perdio la maquina — el checkpoint de la PC conserva el ultimo tramo bajado"; break
  fi
  if printf '%s' "$OUT" | grep -q "VIVO= False"; then LISTO=1; echo "== tramo terminado"; break; fi
done

if [ "$LISTO" = "1" ]; then
  echo "== bajando checkpoint y resultados"
  timeout 600 "${SCP[@]}" "$HOST:$REMOTO/ck.pkl" "$TMP/ck.pkl" >/dev/null 2>&1 \
    && mv "$TMP/ck.pkl" "$CK" && echo "   checkpoint en $CK ($(du -h "$CK" | cut -f1))"
  timeout 300 "${SCP[@]}" "$HOST:$REMOTO/salidas/${UNI}.json" "$JS" >/dev/null 2>&1
  [ -f "$JS" ] && echo "   ultimo paso registrado: $(grep -o '"paso": [0-9]*' "$JS" | tail -1)"
fi
echo "== fin del tramo. La maquina sigue ENCENDIDA y facturando: apagarla desde el panel del proveedor."
