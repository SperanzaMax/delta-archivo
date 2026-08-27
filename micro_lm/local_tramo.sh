#!/usr/bin/env bash
# Corre una unidad en ESTA PC, reanudando desde el checkpoint, con guarda termica activa.
#
#   Uso:  local_tramo.sh <nivel:semilla> <pasos> [cada]
#   Ej.:  local_tramo.sh 4:2 12000 500
#
# La PC es 18-37x mas lenta que una T4 (8,15 s/paso medido), asi que esto NO es la via principal:
# es la red que garantiza avance cuando el pool esta en sequia. Como el checkpoint es el mismo
# archivo que usa Colab, en cuanto una cuenta otorgue GPU se puede matar esto y seguir en la VM
# desde el ultimo multiplo de --cada, sin perder nada.
#
# Tres cuidados sobre la maquina, que es lo que se pidio proteger:
#  1. UN NUCLEO LIBRE. Con los 4 al 100 % la maquina deja de responder (fue asi en la VM de 2 vCPU
#     el 14-ago). Se fija a 3 hilos y se corre con nice 10, asi el escritorio sigue usable.
#  2. TERMICA CON PAUSA, NO CON MUERTE. Un vigilante mira `sensors` cada 30 s: al pasar el techo le
#     manda SIGSTOP al entrenamiento y lo despierta con SIGCONT cuando baja. Pausar no pierde
#     computo; matar perderia hasta --cada pasos. El critico de esta CPU es 100 °C y eso es apagado
#     termico, no advertencia.
#  3. LOCK. Deja ckpts/<uni>.local.lock con el pid para que `rotar_tramos.sh` no tome la misma
#     unidad en Colab y los dos se pisen el checkpoint.
set -uo pipefail

UNIDAD="${1:?falta nivel:semilla}"
PASOS="${2:?faltan los pasos}"
CADA="${3:-500}"
TECHO="${TECHO:-82}"     # pausa por encima de esto
PISO="${PISO:-72}"       # y no reanuda hasta bajar de aca (histeresis: sin esto oscila)

NIVEL="${UNIDAD%%:*}"; SEM="${UNIDAD##*:}"
# Mismos dos ejes que tramo_colab.sh: PREFIJO separa familias que comparten nivel y semilla pero no
# son comparables ("n" = campaña base, "x" = campaña de abstención), y P_NOSE es la proporción de
# preguntas sin respuesta. Sin el prefijo las dos familias escribirían el mismo checkpoint.
PREFIJO="${PREFIJO:-n}"
P_NOSE="${P_NOSE:-0.0}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SALIDA"
CKPTS="$AQUI/ckpts"; mkdir -p "$CKPTS"
UNI="${PREFIJO}${NIVEL}_s${SEM}"
CK="$CKPTS/${UNI}.pkl"
LOCK="$CKPTS/${UNI}.local.lock"
LOG="$AQUI/logs_campania_$(date +%Y%m%d)/local_${UNI}.log"
mkdir -p "$(dirname "$LOG")"
PY=/home/maxi/.venv-ligamento/bin/python

. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo
mandar() { curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
             -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1; }

temp() {
  sensors 2>/dev/null | sed 's/(.*//' | grep -oE '\+[0-9]+\.[0-9]+°C' \
    | tr -d '+°C' | sort -rn | head -1 | cut -d. -f1
}

if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "ya hay una corrida local de $UNI (pid $(cat "$LOCK")); no se arranca otra"; exit 1
fi
[ -x "$PY" ] || { echo "no esta el interprete $PY"; exit 1; }
"$AQUI/termica.sh" esperar || { echo "la maquina no baja de temperatura; no se arranca"; exit 1; }

echo "== local · $UNI · hasta el paso $PASOS · ckpt $CK · techo ${TECHO} °C"
# OMP_NUM_THREADS NO ALCANZA (medido el 17-ago, con el escritorio de Maxi congelandose): XLA arma su
# propio pool de hilos para el backend de CPU y no mira esa variable. El proceso quedaba al 256 % con
# load 3,96 sobre 4 nucleos y habia que mover el mouse para destrabar la sesion grafica. Lo que si
# funciona es AFINIDAD: `taskset` le prohibe el ultimo nucleo, que queda entero para gnome-shell y
# Xorg. Se pierde ~25 % de velocidad en una corrida que de todos modos es la red y no la via
# principal; a cambio la maquina sigue siendo usable, que es la condicion que puso Maxi.
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export JAX_PLATFORMS=cpu
# Dos palancas distintas, y hacen falta las DOS (2026-08-17, a pedido de Maxi: «2/3 núcleos sólo al
# 60 %»). `taskset` elige CUANTOS nucleos, pero no limita cuanto los usa: con afinidad sola el
# proceso los deja al 100 % y el escritorio se traba igual. La cuota va por cgroup v2, que en esta
# maquina esta DELEGADO al usuario (`cpu` aparece en cgroup.controllers de user@1000.service), asi
# que no hace falta sudo ni cpulimit —que no esta instalado—.
NUCLEOS="${NUCLEOS:-0-2}"
PCT="${PCT:-60}"          # % de cada nucleo
CG_BASE=/sys/fs/cgroup/user.slice/user-1000.slice/user@1000.service
CG="$CG_BASE/microlm"

# cpu.max es "<cuota> <periodo>" en microsegundos: la cuota es el tiempo de CPU que el cgroup puede
# gastar en cada periodo. 3 nucleos al 60 % = 180 % de un nucleo = 180000 sobre 100000.
limitar_cpu() {
  local pid="$1" n_cpus cuota
  n_cpus=$(echo "$NUCLEOS" | awk -F- '{print ($2 ? $2-$1+1 : 1)}')
  cuota=$(( n_cpus * PCT * 1000 ))
  mkdir -p "$CG" 2>/dev/null || { echo "   (sin cgroup delegado: sigue sin cuota)"; return; }
  echo "$cuota 100000" > "$CG/cpu.max" 2>/dev/null || { echo "   (no se pudo fijar cpu.max)"; return; }
  # Los HILOS van uno por uno: mover el pid solo deja el resto donde estaba.
  for t in /proc/"$pid"/task/*; do echo "$(basename "$t")" > "$CG/cgroup.threads" 2>/dev/null; done
  echo "$pid" > "$CG/cgroup.procs" 2>/dev/null
  echo "   cuota de CPU: ${n_cpus} nucleos x ${PCT}% = $((n_cpus * PCT))% total"
}

taskset -c "$NUCLEOS" nice -n 15 "$PY" -u "$AQUI/entrenar.py" \
    --nivel "$NIVEL" --semilla "$SEM" --pasos "$PASOS" --cada "$CADA" \
    --d 128 --capas 4 --lr 1e-3 --p-vieja 0.35 --idioma 2 --horizonte 20000 \
    --p-nose "$P_NOSE" \
    --salida "$SALIDA/${UNI}.json" --ckpt "$CK" >> "$LOG" 2>&1 &
ENTREN=$!
echo "$ENTREN" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT
echo "   pid $ENTREN · log $LOG"
sleep 5   # que JAX termine de crear su pool de hilos antes de moverlos al cgroup
limitar_cpu "$ENTREN"
mandar "🖥 micro-LM · $UNI arranca EN LA PC desde el paso $(grep -o '\"paso\": [0-9]*' "$SALIDA/${UNI}.json" 2>/dev/null | tail -1 | grep -o '[0-9]*' || echo '?').
Ritmo ~8,2 s/paso. Guarda térmica: pausa sobre ${TECHO} °C, sigue bajo ${PISO} °C."

PAUSADO=0
while kill -0 "$ENTREN" 2>/dev/null; do
  T="$(temp)"
  if [ -n "$T" ]; then
    if [ "$PAUSADO" = "0" ] && [ "$T" -ge "$TECHO" ]; then
      kill -STOP "$ENTREN" 2>/dev/null && PAUSADO=1
      echo "   [$(date +%H:%M:%S)] $T °C ≥ $TECHO → PAUSA"
      mandar "🌡 micro-LM · $UNI en pausa: la CPU llegó a $T °C (techo $TECHO). Sigue sola al bajar de $PISO."
    elif [ "$PAUSADO" = "1" ] && [ "$T" -le "$PISO" ]; then
      kill -CONT "$ENTREN" 2>/dev/null && PAUSADO=0
      echo "   [$(date +%H:%M:%S)] $T °C ≤ $PISO → SIGUE"
      mandar "🌡 micro-LM · $UNI reanuda: la CPU bajó a $T °C."
    fi
  fi
  sleep 30
done

rm -f "$LOCK"
P="$(grep -o '"paso": [0-9]*' "$SALIDA/${UNI}.json" 2>/dev/null | tail -1 | grep -o '[0-9]*')"
echo "== local $UNI termino · ultimo paso ${P:-?}"
mandar "🖥 micro-LM · la corrida local de $UNI terminó en el paso ${P:-?} de $PASOS."
