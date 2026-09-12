#!/usr/bin/env bash
# CAMPANIA HECHOS ENCADENADOS v4 · PREREG_ENCADENADOS.md §8 (E-4) · 2026-09-12 11:20 · bloque barajado y de tres
#   nc3_sX  lectura en el bloque 0        (H0': con el bloque de altura en otra sesion, no puede)
#   nd3_sX  lectura en los bloques 0 y 2   (H1': si)
# Las cuatro sembradas de kc3_s1 / kc3_s2 (`sembrar.py --horizonte 8000`), mismos datos.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=abs PERT=0 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0 P_COMPUESTA=0.5
       REL2_SESION=1 REL2_BARAJAR=1 FOTOS=100 HORIZONTE=8000 VUELTAS=12)
cd "$AQUI"
for u in nc3_s1 nc3_s2 nd3_s1 nd3_s2; do
  [ -f "ckpts/$u.pkl" ] || { echo "falta ckpts/$u.pkl (sembrar primero)"; exit 1; }
done
uno_rot() {  # familia bloques unidad cuentas...
  local fam="$1" bl="$2" u="$3"; shift 3
  env "${COMUN[@]}" PREFIJO="$fam" BLOQUES_LECTURA="$bl" LOG_ROTADOR="$SAL/rotador_${fam}3_s${u##*:}.log" \
    setsid nohup ./rotar_abst3.sh "$u" 8000 2000 500 "$@" > "$SAL/rotador_${fam}3_s${u##*:}.log" 2>&1 < /dev/null &
  echo "rotador ${fam}3_s${u##*:} pid $! cuentas $*"; sleep 3
}
uno_rot nc 0 3:1 N F
uno_rot nc 0 3:2 G J
uno_rot nd 0,2 3:1 C L
uno_rot nd 0,2 3:2 D I
