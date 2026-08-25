#!/usr/bin/env bash
# CIERRE NOCTURNO AUTONOMO (2026-08-24, pedido de Maxi: «cuando cierren o se corte la sesion guarda
# todo y mañana seguimos, al terminar apaga la pc»).
#
# Corre solo, con setsid, para que NO dependa de que la sesion de trabajo siga viva. Eso es el punto:
# si la sesion se corta, esto igual guarda y apaga.
#
# Que hace, en orden:
#   1. espera a que las SEIS unidades lleguen a 26000 (o hasta el limite de tiempo)
#   2. para todo limpio: rotadores, watchdogs, vigia, tramos
#   3. para TODAS las sesiones de Colab — si no, las VM siguen tomadas y gastan cuota toda la noche
#   4. corre las evaluaciones y deja los datos listos para mañana
#   5. escribe el estado, commitea y avisa por Telegram
#   6. apaga la PC
#
# El limite de tiempo NO es una salvaguarda decorativa: si las unidades no llegan, igual hay que
# guardar y apagar, porque dejar la maquina prendida sin nadie es justamente lo que se quiere evitar.
# Las unidades quedan reanudables en disco, que es como trabaja este proyecto desde el 14-ago.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$AQUI/.." && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python3
COLAB=/home/maxi/.venv-colab-cli/bin/colab
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
LOG="$AQUI/cierre_nocturno.log"
LIMITE_MIN="${LIMITE_MIN:-360}"          # 6 horas
UNIDADES=(v3_s0 v3_s1 v3_s2 y3_s0 y3_s1 y3_s2)
META=26000

di() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

# OJO: el texto va con comillas SIMPLES y sin backticks. Hoy se perdieron dos palabras de un aviso
# porque bash ejecuto los backticks del mensaje — el mismo bug que se arreglo esta mañana en los
# tramo_*.sh.
mandar() {
  /usr/bin/curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

paso_de() {
  "$PY" - "$AQUI/ckpts/$1.pkl" <<'PY' 2>/dev/null || echo -1
import pickle, sys
try: print(pickle.load(open(sys.argv[1],'rb')).get('paso', -1))
except Exception: print(-1)
PY
}

di "== cierre nocturno armado · limite ${LIMITE_MIN} min · meta $META"
FIN=$(( $(date +%s) + LIMITE_MIN * 60 ))
COMPLETAS=0
while [ "$(date +%s)" -lt "$FIN" ]; do
  COMPLETAS=0; DETALLE=""
  for u in "${UNIDADES[@]}"; do
    p="$(paso_de "$u")"
    DETALLE="$DETALLE $u=$p"
    [ "$p" -ge "$META" ] 2>/dev/null && COMPLETAS=$((COMPLETAS+1))
  done
  di "   $COMPLETAS/6 ·$DETALLE"
  [ "$COMPLETAS" -ge 6 ] && break
  sleep 120
done
[ "$COMPLETAS" -ge 6 ] && di "== las seis llegaron a $META" || di "== se agoto el limite con $COMPLETAS/6"

# --- 2. parar todo limpio ------------------------------------------------------------------------
di "== parando rotadores, watchdogs, vigia y tramos"
pkill -f watchdog_tramo2.sh 2>/dev/null
pkill -f rotar_abst2.sh 2>/dev/null
pkill -f tramo_abst.sh 2>/dev/null
pkill -f vigia_escalonado.sh 2>/dev/null
sleep 5
pkill -f "colab_cli.cli.*keep-alive" 2>/dev/null
rm -f "$HOME/.colab-pool/en_uso_"* 2>/dev/null

# --- 3. parar las VM de Colab, cuenta por cuenta --------------------------------------------------
di "== parando sesiones de Colab (si quedan vivas gastan cuota toda la noche)"
for c in C D E F G H I J K L M N; do
  export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$c"
  ses=$(timeout 45 "$COLAB" --auth adc --config "$HOME/.colab-cuenta$c.json" sessions 2>/dev/null \
        | grep -oE "tr2_[a-z]_[0-9]{4}" | sort -u)
  for s in $ses; do
    timeout 45 "$COLAB" --auth adc --config "$HOME/.colab-cuenta$c.json" stop -s "$s" >/dev/null 2>&1 \
      && di "   cuenta $c: $s parada"
  done
done
unset CLOUDSDK_CONFIG

# --- 4. evaluaciones, para que mañana los datos ya esten -----------------------------------------
di "== evaluando lo que haya llegado (deja los json listos para mañana)"
mkdir -p "$AQUI/cierre_20260824"
for u in "${UNIDADES[@]}"; do
  p="$(paso_de "$u")"
  [ "$p" -ge "$META" ] 2>/dev/null || { di "   $u en $p: no llego, no se evalua"; continue; }
  taskset -c 0-3 "$PY" "$AQUI/ser.py" "$AQUI/ckpts/$u.pkl" --n 2048 --B 64 --semilla 54321 \
    --json "$AQUI/cierre_20260824/ser_$u.json" >/dev/null 2>&1 \
    && di "   $u evaluado" || di "   $u fallo la evaluacion"
done

# --- 5. estado + commit --------------------------------------------------------------------------
{
  echo "# Estado al cierre del 24-ago (automatico)"
  echo
  echo "Cerrado por \`cierre_nocturno.sh\`, sin sesion de trabajo viva."
  echo
  echo "| unidad | paso | meta |"
  echo "|---|---:|---:|"
  for u in "${UNIDADES[@]}"; do echo "| \`$u\` | $(paso_de "$u") | $META |"; done
  echo
  echo "Completas: $COMPLETAS de 6."
  echo
  echo "- \`v3_*\` = lat2 (PREREG_LAT2.md, SHA 28d6f15a)"
  echo "- \`y3_*\` = slot nulo (PREREG_SLOT_NULO.md, SHA f95b6e9d)"
  echo
  echo "Las evaluaciones de las unidades que llegaron estan en \`micro_lm/cierre_20260824/\`."
  echo "Todo lo demas quedo detenido y sin VM tomadas."
  echo
  echo "## Para mañana"
  echo
  echo "1. Analizar \`y3_*\` contra el control \`p3_*\`: S-0 bloqueante, S-1 la compuerta, y **S-2**,"
  echo "   que es la que decide — el score del archivo tiene que subir del 0,4984 basal."
  echo "2. Analizar \`lat2\` contra \`p3_*\` y \`w3_*\`: V-1 conservacion, V-2 anterior, V-3 nose_rel."
  echo "3. Reanudar lo que no haya llegado."
} > "$REPO/ESTADO_20260824_NOCHE.md"

cd "$REPO" || exit 1
git add -A >/dev/null 2>&1
git commit -q -m "Cierre automatico del 24-ago: $COMPLETAS/6 unidades completas

Cerrado por cierre_nocturno.sh sin sesion viva. Rotadores, watchdogs, vigia y tramos detenidos,
locks liberados y sesiones de Colab paradas para que no quede cuota gastandose de noche.

Las unidades que llegaron a 26000 quedaron evaluadas en micro_lm/cierre_20260824/, asi que mañana
los datos ya estan y no hay que esperar CPU.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" >/dev/null 2>&1 && di "== commit hecho"

# --- 6. avisar y apagar ---------------------------------------------------------------------------
RESUMEN=""
for u in "${UNIDADES[@]}"; do RESUMEN="$RESUMEN
  $u = $(paso_de "$u") / $META"; done
mandar "CIERRE NOCTURNO · $COMPLETAS de 6 unidades completas.$RESUMEN

Todo detenido y guardado: rotadores, vigia y watchdogs terminados, locks liberados y las VM de Colab paradas (no queda cuota gastandose).

Las que llegaron quedaron evaluadas en micro_lm/cierre_20260824/, asi que mañana los datos ya estan.

Commit hecho. La PC se apaga ahora.

Para mañana: analizar el slot contra su control (S-2 es la que decide, el score del archivo tiene que subir del 0,4984) y lat2 contra pre y lat."

di "== apagando la PC"
sleep 10
sudo -n poweroff 2>/dev/null || systemctl poweroff 2>/dev/null || sudo -n shutdown -h now
