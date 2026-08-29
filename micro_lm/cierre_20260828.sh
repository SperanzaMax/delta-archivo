#!/usr/bin/env bash
# Cierre ordenado de la campania PREREG_TASA_REGIMEN, noche del 28-ago-2026.
#
# El orden importa y sale de dos incidentes de HOY:
#   1. Matar el rotador NO alcanza: sus `tramo_abst.sh` sobreviven reparentados y siguen bajando
#      checkpoints de la VM vieja, pisando lo que escriba el rotador siguiente. Se barren los hijos.
#   2. Los locks de ~/.colab-pool quedan tomados si el rotador no cierra solo, y bloquean la cuenta
#      la proxima vez. Se limpian los que apunten a un PID muerto.
#
# Las sesiones de Colab se paran explicitamente: dejarlas vivas consume cuota de las cuentas del pool
# toda la noche sin entrenar nada.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python
COLAB=/home/maxi/.venv-colab-cli/bin/colab

echo "== CIERRE $(date +%H:%M) =="

# ---- 1. Sesiones vivas, ANTES de matar nada (despues no sabriamos cuales eran)
SESIONES=$(ps -eo args | grep -oE "tr2_[a-z]_[0-9]{4}" | sort -u)
echo "-- sesiones de Colab detectadas: ${SESIONES:-ninguna}"

# ---- 2. Parar rotadores, y despues sus hijos
for p in $(pgrep -f "rotar_abst3.sh 3:" 2>/dev/null); do
  echo "-- rotador $p"; kill -TERM "$p" 2>/dev/null
done
sleep 8
for p in $(pgrep -f "tramo_abst.sh" 2>/dev/null); do
  echo "-- tramo huerfano $p"; kill -TERM "$p" 2>/dev/null
done
for p in $(pgrep -f "watchdog_tramo2.sh" 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
sleep 5
for p in $(pgrep -f "colab_cli.cli.*keep-alive" 2>/dev/null); do kill -TERM "$p" 2>/dev/null; done
sleep 3
pkill -KILL -f "tramo_abst.sh" 2>/dev/null
pkill -KILL -f "rotar_abst3.sh 3:" 2>/dev/null

# ---- 3. Parar las VMs para no gastar cuota toda la noche
for s in $SESIONES; do
  echo "-- stop $s"; timeout 90 "$COLAB" --auth adc stop -s "$s" >/dev/null 2>&1 && echo "   ok" || echo "   no respondio (se recicla sola)"
done

# ---- 4. Locks huerfanos
for l in "$HOME"/.colab-pool/en_uso_*; do
  [ -e "$l" ] || continue
  pid=$(cat "$l" 2>/dev/null)
  if ! kill -0 "$pid" 2>/dev/null; then echo "-- lock huerfano $(basename "$l")"; rm -f "$l"; fi
done

# ---- 5. Estado final de cada unidad
echo
echo "== PASOS AL CIERRE =="
for u in 0 1 2 3 4 5 6 7 8; do
  ck="$AQUI/ckpts/b3_s$u.pkl"
  [ -f "$ck" ] || continue
  p=$("$PY" -c "import pickle;print(pickle.load(open('$ck','rb')).get('paso') or 0)" 2>/dev/null)
  printf "   b3_s%-2s %6s / 26000\n" "$u" "$p"
done

echo
echo "== procesos que quedan vivos =="
pgrep -af "rotar_abst3|tramo_abst|keep-alive" 2>/dev/null || echo "   ninguno"
echo "== CIERRE COMPLETO $(date +%H:%M) =="
