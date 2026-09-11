#!/usr/bin/env bash
# CAMPANIA tf3/df3 · PREREG_TOPK_ENTRENADO.md §9 · 2026-09-11 10:20
# Archivo largo FRESCO: 36 sesiones extra escritas con los pesos actuales, sin gradiente.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=rel PERT=1 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0
       SES_EXTRA=36 SES_EXTRA_SIN_GRAD=1 RELLENO=0 FOTOS=25 HORIZONTE=8000 VUELTAS=12)
cd "$AQUI"
uno_rot() {  # familia topk unidad cuentas...
  local fam="$1" k="$2" u="$3"; shift 3
  env "${COMUN[@]}" PREFIJO="$fam" TOPK="$k" LOG_ROTADOR="$SAL/rotador_${fam}3_s${u##*:}.log" \
    setsid nohup ./rotar_abst3.sh "$u" 8000 2000 500 "$@" > "$SAL/rotador_${fam}3_s${u##*:}.log" 2>&1 < /dev/null &
  echo "rotador ${fam}3_s${u##*:} pid $! cuentas $*"; sleep 3
}
case "${1:-}" in
  s0) uno_rot tf 2 3:0 K C; uno_rot df 0 3:0 L D ;;
  s12) uno_rot tf 2 3:1 N I; uno_rot tf 2 3:2 F J; uno_rot df 0 3:1 E A; uno_rot df 0 3:2 H M ;;
  *) echo "uso: $0 s0 | s12" ;;
esac
