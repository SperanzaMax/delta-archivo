#!/usr/bin/env bash
# DIAGNOSTICO de la divergencia de tf3/df3 · 2026-09-11 12:05
#   xa  ses_extra 12 SIN grad · cabeza · p_nose 0.4    (= tf3, mas chico)
#   xb  ses_extra 12 CON grad · cabeza · p_nose 0.4    (aisla el stop_gradient)
#   xc  ses_extra 12 SIN grad · token  · p_nose 0.2    (aisla la cabeza)
#   xd  ses_extra 12 CON grad · token  · p_nose 0.2    (= rp3 con 12; control positivo)
# 600 pasos, eval cada 100, sin fotos, sin topk. Todo sembrado de kq3_s0.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=rel PERT=1 KERNEL_Q=5 DONDE=lat2 SEMBRAR=0 SES_EXTRA=12 RELLENO=0 FOTOS=0 HORIZONTE=600 VUELTAS=6 TOPK=0 MICRO_BATCH=8)
cd "$AQUI"
uno() {  # prefijo cuentas... (con las variables extra en el env del llamador)
  local fam="$1"; shift
  env "${COMUN[@]}" PREFIJO="$fam" LOG_ROTADOR="$SAL/rotador_${fam}3_s0.log" \
    setsid nohup ./rotar_abst3.sh 3:0 600 600 100 "$@" > "$SAL/rotador_${fam}3_s0.log" 2>&1 < /dev/null &
  echo "rotador ${fam}3_s0 pid $! cuentas $*"; sleep 3
}
SES_EXTRA_SIN_GRAD=1 ABST=cabeza P_NOSE=0.4 uno xa K C
SES_EXTRA_SIN_GRAD=0 ABST=cabeza P_NOSE=0.4 uno xb L D
SES_EXTRA_SIN_GRAD=1 ABST=token  P_NOSE=0.2 uno xc N I
SES_EXTRA_SIN_GRAD=0 ABST=token  P_NOSE=0.2 uno xd F J
