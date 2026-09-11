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
if [ "${1:-}" = "" ]; then
uno_rot tt 2 3:0 K C
uno_rot tt 2 3:1 L D
uno_rot tt 2 3:2 N I
uno_rot rq 0 3:0 F J
fi

# --- 14:50 · segunda tanda de cuentas. Las cuatro VMs se cayeron dos veces a la vez (13:37 y 14:45) y
# las listas de dos cuentas se agotaron. Se relanza con cuentas que hoy descansaron mas; los
# checkpoints de la PC tienen el paso (1.250 / 1.500 / 1.250 / 1.500) y SEMBRAR=0 reanuda.
if [ "${1:-}" = "relanzar" ]; then
  uno_rot tt 2 3:0 A E
  uno_rot tt 2 3:1 G H
  uno_rot tt 2 3:2 M K
  uno_rot rq 0 3:0 C L
fi

# --- P4 (17:05): 52 sesiones viejas con gradiente, micro-batch 4.
if [ "${1:-}" = "p4" ]; then
  env "${COMUN[@]}" PREFIJO=tw TOPK=2 SES_EXTRA=52 MICRO_BATCH=4 BATCH_EVAL=4 MIN_POR_MIL=40 LOG_ROTADOR="$SAL/rotador_tw3_s0.log" \
    setsid nohup ./rotar_abst3.sh 3:0 2000 1000 250 E G > "$SAL/rotador_tw3_s0.log" 2>&1 < /dev/null &
  echo "rotador tw3_s0 pid $! cuentas E G"
fi
