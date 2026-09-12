#!/usr/bin/env bash
# CAMPANIA HECHOS ENCADENADOS v3 · PREREG_ENCADENADOS.md §7 (E-3) · 2026-09-12 09:25
#   mc3_sX  lectura en el bloque 0        (H0': con el bloque de altura en otra sesion, no puede)
#   md3_sX  lectura en los bloques 0 y 2   (H1': si)
# Las cuatro sembradas de kc3_s1 / kc3_s2 (`sembrar.py --horizonte 8000`), mismos datos.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=abs PERT=0 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0 P_COMPUESTA=0.5
       REL2_SESION=1 FOTOS=100 HORIZONTE=8000 VUELTAS=12)
cd "$AQUI"
for u in mc3_s1 mc3_s2 md3_s1 md3_s2; do
  [ -f "ckpts/$u.pkl" ] || { echo "falta ckpts/$u.pkl (sembrar primero)"; exit 1; }
done
uno_rot() {  # familia bloques unidad cuentas...
  local fam="$1" bl="$2" u="$3"; shift 3
  env "${COMUN[@]}" PREFIJO="$fam" BLOQUES_LECTURA="$bl" LOG_ROTADOR="$SAL/rotador_${fam}3_s${u##*:}.log" \
    setsid nohup ./rotar_abst3.sh "$u" 8000 2000 500 "$@" > "$SAL/rotador_${fam}3_s${u##*:}.log" 2>&1 < /dev/null &
  echo "rotador ${fam}3_s${u##*:} pid $! cuentas $*"; sleep 3
}
uno_rot mc 0   3:1 C L
uno_rot mc 0   3:2 D N
uno_rot md 0,2 3:1 I F
uno_rot md 0,2 3:2 J G
