#!/usr/bin/env bash
# Campania del MICRO-LM: 4 niveles x 3 semillas = 12 corridas, repartidas en 8 cuentas de Colab.
#
#   Uso:  campania_micro.sh [pasos]
#
# Por que 3 semillas: E-I3d dejo `ANTERIOR` BIMODAL (0,0052 · 0,9297 · 0,0078) — dos semillas
# cayeron en el atajo de la recencia y una aprendio la operacion completa. Con una sola semilla ese
# resultado se lee como «anda» o «no anda» segun cual toque. **Se reporta POR SEMILLA, nunca solo la
# media.**
#
# El reparto esparce cada nivel entre 3 cuentas distintas, para que ninguna diferencia entre cuentas
# caiga adentro del contraste que se quiere medir (el contraste es el NIVEL).
set -uo pipefail

PASOS="${1:-20000}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$AQUI/logs_campania_$(date +%Y%m%d)"
mkdir -p "$LOGS"

# cuenta -> unidades (nivel:semilla). Cada nivel aparece en 3 cuentas distintas.
declare -A REPARTO=(
  [A]="1:0,2:1"   [C]="2:0,3:1"   [D]="3:0,4:1"   [E]="4:0,1:1"
  [F]="1:2"       [G]="2:2"       [H]="3:2"       [I]="4:2"
)

echo "== campania micro-LM · $PASOS pasos · 12 unidades en 8 cuentas"
for c in A C D E F G H I; do
  echo "   $c -> ${REPARTO[$c]}"
done
echo

for c in A C D E F G H I; do
  # Una cuenta = un proceso `colab`, nunca dos: dos comandos concurrentes sobre el mismo estado se
  # pisan el sessions.json y dejan la sesion inalcanzable por CLI (paso el 2026-08-09, costo 2 VMs).
  nohup "$AQUI/lanzar_micro_colab.sh" "$c" "${REPARTO[$c]}" "$PASOS" \
      > "$LOGS/$c.log" 2>&1 &
  echo "lanzada cuenta $c (pid $!) -> $LOGS/$c.log"
  sleep 10          # escalonado: 8 `colab new` simultaneos se pisan en el backend
done

echo
echo "todas lanzadas. Seguimiento:"
echo "  tail -f $LOGS/*.log"
echo "  ls $AQUI/corridas_$(date +%Y%m%d)/"
wait
echo "== campania terminada"
