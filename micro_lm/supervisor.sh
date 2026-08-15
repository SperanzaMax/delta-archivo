#!/usr/bin/env bash
# Espera a que se liberen las cuentas que HOY tienen GPU y les pasa las unidades pendientes.
#
#   Uso:  supervisor.sh
#
# Por que (2026-08-14): la cuota de GPU es POR CUENTA. J y E no consiguen ni T4 ni L4 ni A100 —o sea
# se quedaron sin cuota, no es escasez del momento—, mientras F, G y H llevan mas de una hora
# corriendo sin un solo tropiezo. Insistir con las cuentas secas es tiempo perdido: lo que hay que
# hacer es esperar a que las que SI andan terminen su unidad y darles la siguiente.
#
# Mata los rotadores que esten dando vueltas sobre cuentas secas antes de relanzar, para no tener
# dos procesos `colab` sobre la misma cuenta.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$AQUI/logs_campania_$(date +%Y%m%d)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
PASOS="${1:-20000}"
BUENAS="${2:-F G H}"

pendientes() {
  local out=()
  for n in 1 2 3 4; do
    for s in 0 1 2; do
      local f="$SALIDA/n${n}_s${s}.json"
      [ -f "$f" ] && grep -q "\"paso\": $PASOS" "$f" 2>/dev/null && continue
      out+=("$n:$s")
    done
  done
  (IFS=,; echo "${out[*]}")
}

echo "== supervisor · espera a que se liberen: $BUENAS"
while true; do
  ocupadas=0
  for c in $BUENAS; do
    pgrep -f "lanzar_micro[0-9]*\.sh $c " >/dev/null 2>&1 && ocupadas=$((ocupadas + 1))
  done
  [ "$ocupadas" -eq 0 ] && break
  sleep 60
done

echo "== las cuentas buenas se liberaron; matando rotadores sobre cuentas secas"
pkill -9 -f "rotar_cuentas.sh" 2>/dev/null
sleep 3
pkill -9 -f "lanzar_micro[23]\.sh [JEDAIC] " 2>/dev/null
sleep 3

FALTAN="$(pendientes)"
if [ -z "$FALTAN" ]; then echo "== no falta nada"; exit 0; fi
echo "== pendientes: $FALTAN · repartiendo entre $BUENAS"

# Reparto DISJUNTO: si a las tres cuentas se les diera la lista entera, las tres empezarian por la
# misma unidad y dos de cada tres corridas serian tiradas a la basura. Se reparte por turno
# (unidad k -> cuenta k mod n), que ademas deja cada nivel esparcido entre cuentas distintas.
read -r -a ARR_U <<< "$(echo "$FALTAN" | tr ',' ' ')"
read -r -a ARR_C <<< "$BUENAS"
declare -A LOTE
for k in "${!ARR_U[@]}"; do
  c="${ARR_C[$((k % ${#ARR_C[@]}))]}"
  LOTE[$c]="${LOTE[$c]:+${LOTE[$c]},}${ARR_U[$k]}"
done
for c in "${ARR_C[@]}"; do
  [ -z "${LOTE[$c]:-}" ] && continue
  nohup "$AQUI/rotar_cuentas.sh" "${LOTE[$c]}" "$PASOS" "$c" >> "$LOGS/sup_${c}.log" 2>&1 &
  echo "   $c -> ${LOTE[$c]} (pid $!)"
  sleep 20
done
wait
echo "== supervisor fin · pendientes: $(pendientes)"
