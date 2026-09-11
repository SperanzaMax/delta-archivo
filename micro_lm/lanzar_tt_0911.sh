#!/usr/bin/env bash
# CAMPANIA tt3 (rp3 + top-2) y rq3 (replica de rp3_s0 con fotos) · PREREG_TOPK_ENTRENADO.md §10 · 2026-09-11 12:50
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=rel PERT=1 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.2 ABST=token SEMBRAR=0 SES_EXTRA=26 SES_EXTRA_SIN_GRAD=0
       MICRO_BATCH=8 BATCH_EVAL=8 RELLENO=0 FOTOS=8 HORIZONTE=6000 VUELTAS=12 MIN_POR_MIL=20)
cd "$AQUI"
uno_rot() {  # familia topk unidad cuentas...
  local fam="$1" k="$2" u="$3"; shift 3
  env "${COMUN[@]}" PREFIJO="$fam" TOPK="$k" LOG_ROTADOR="$SAL/rotador_${fam}3_s${u##*:}.log" \
    setsid nohup ./rotar_abst3.sh "$u" 2000 2000 250 "$@" > "$SAL/rotador_${fam}3_s${u##*:}.log" 2>&1 < /dev/null &
  echo "rotador ${fam}3_s${u##*:} pid $! cuentas $*"; sleep 3
}
uno_rot tt 2 3:0 K C
uno_rot tt 2 3:1 L D
uno_rot tt 2 3:2 N I
uno_rot rq 0 3:0 F J
