#!/usr/bin/env bash
# Corre un conjunto de unidades ROTANDO entre cuentas hasta que alguna consiga una T4.
#
#   Uso:  rotar_cuentas.sh <unidades> [pasos] [cuentas...]
#   Ej.:  rotar_cuentas.sh 1:0,3:0 20000 J D E A C I
#
# Por que existe (2026-08-14, observacion de Maxi): las cuentas se cuelgan de a ratos. Hoy la
# primera tanda consiguio ocho T4 a las 08:10 y para las 12:30 ninguna de J/D/E/I lograba
# asignacion, con 503 «Service Unavailable» sostenido, mientras F/G/H seguian corriendo sin
# problema. Insistir en la MISMA cuenta es la estrategia equivocada: lo que funciona es probar la
# siguiente.
#
# Se apoya en que `lanzar_micro_colab.sh` saltea las unidades ya completas (`pendientes()`), asi que
# llamarlo de nuevo con otra cuenta REANUDA lo que falta en vez de repetir lo hecho. Cada cuenta se
# prueba con MAX_INTENTOS=1 para rotar rapido en lugar de insistir seis minutos por cuenta.
set -uo pipefail

UNIDADES="${1:?faltan las unidades, p.ej. 1:0,3:0}"
PASOS="${2:-20000}"
shift 2 2>/dev/null || shift 1
CUENTAS=("${@:-J D E A C I H G F}")
[ "$#" -eq 0 ] && CUENTAS=(J D E A C I H G F)

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
LOGS="$AQUI/logs_campania_$(date +%Y%m%d)"
mkdir -p "$SALIDA" "$LOGS"

pendientes() {
  local out=()
  for u in $(echo "$UNIDADES" | tr ',' ' '); do
    local n="${u%%:*}" s="${u##*:}"
    local f="$SALIDA/n${n}_s${s}.json"
    [ -f "$f" ] && grep -q "\"paso\": $PASOS" "$f" 2>/dev/null && continue
    out+=("$u")
  done
  (IFS=,; echo "${out[*]}")
}

echo "== rotador · unidades $UNIDADES · cuentas: ${CUENTAS[*]}"
for vuelta in 1 2 3; do
  for c in "${CUENTAS[@]}"; do
    FALTAN="$(pendientes)"
    if [ -z "$FALTAN" ]; then echo "== todo completo"; exit 0; fi
    echo "-- vuelta $vuelta · probando cuenta $c · faltan $FALTAN"
    MAX_INTENTOS=1 "$AQUI/lanzar_micro3.sh" "$c" "$FALTAN" "$PASOS" \
        >> "$LOGS/rot_${c}.log" 2>&1
    echo "   cuenta $c devolvio; pendientes ahora: $(pendientes)"
  done
  echo "-- fin de la vuelta $vuelta; espera 3 min antes de reintentar el pool"
  sleep 180
done
echo "== rotador fin · pendientes: $(pendientes)"
