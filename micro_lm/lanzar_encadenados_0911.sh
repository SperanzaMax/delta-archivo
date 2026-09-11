#!/usr/bin/env bash
# CAMPANIA HECHOS ENCADENADOS · PREREG_ENCADENADOS.md · 2026-09-11 18:55
#   ec3_sX  lectura en el bloque 0        (H0: no puede encadenar)
#   ed3_sX  lectura en los bloques 0 y 2   (H1: si)
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=abs PERT=0 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0 P_COMPUESTA=0.5
       FOTOS=100 HORIZONTE=26000 VUELTAS=12)
cd "$AQUI"
uno_rot() {  # familia bloques unidad cuentas...
  local fam="$1" bl="$2" u="$3"; shift 3
  env "${COMUN[@]}" PREFIJO="$fam" BLOQUES_LECTURA="$bl" LOG_ROTADOR="$SAL/rotador_${fam}3_s${u##*:}.log" \
    setsid nohup ./rotar_abst3.sh "$u" 26000 2000 1000 "$@" > "$SAL/rotador_${fam}3_s${u##*:}.log" 2>&1 < /dev/null &
  echo "rotador ${fam}3_s${u##*:} pid $! cuentas $*"; sleep 3
}
uno_rot kc 0   3:0 K C
uno_rot kc 0   3:1 L D
uno_rot kc 0   3:2 N I
uno_rot kd 0,2 3:0 F J
uno_rot kd 0,2 3:1 H M
uno_rot kd 0,2 3:2 A G
