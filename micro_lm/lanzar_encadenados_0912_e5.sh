#!/usr/bin/env bash
# ENCADENADOS E-5 · PREREG_ENCADENADOS.md §10 · 2026-09-12 10:55
#   nd3_s3-s6  lat2, bloques 0,2, sembradas de kc3_s2 con otra semilla (tasa de arranque de la segunda lectura)
#   ne3_s1-s2  attn, bloques 0,2, sembradas de kc3_s2 (la query global como tercera arquitectura)
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=abs PERT=0 KERNEL_Q=5 P_NOSE=0.4 ABST=cabeza SEMBRAR=0 P_COMPUESTA=0.5
       REL2_SESION=1 REL2_BARAJAR=1 BLOQUES_LECTURA=0,2 FOTOS=100 HORIZONTE=8000 VUELTAS=${VUELTAS:-12})
cd "$AQUI"
uno_rot() {  # familia donde unidad cuentas...
  local fam="$1" dd="$2" u="$3"; shift 3
  [ -f "ckpts/${fam}3_s${u##*:}.pkl" ] || { echo "falta ckpts/${fam}3_s${u##*:}.pkl"; return; }
  env "${COMUN[@]}" PREFIJO="$fam" DONDE="$dd" LOG_ROTADOR="$SAL/rotador_${fam}3_s${u##*:}.log" \
    setsid nohup ./rotar_abst3.sh "$u" 8000 2000 500 "$@" > "$SAL/rotador_${fam}3_s${u##*:}.log" 2>&1 < /dev/null &
  echo "rotador ${fam}3_s${u##*:} pid $! cuentas $*"; sleep 3
}
# --- 15:55 · relanzar lo pendiente. Las 12 vueltas se agotaron sin T4 a media tarde; nd3_s4 y nd3_s6
# cerraron (8000), nd3_s5 quedo en 6000, nd3_s3 en 2000, y ne3 NUNCA corrio: la guarda de `donde` en
# entrenar.py resucitaba `pre` sobre el checkpoint sembrado (arreglado, con smoke local).
# --- 21:15 · E-7 (§14): pesos propios para la segunda lectura, sembradas de kc3_s2 (la base saturada).
if [ "${1:-}" = "e7" ]; then
  uno_rot ng lat2 3:2 N F G
  uno_rot ng lat2 3:3 C L D
  uno_rot ng lat2 3:4 M K A
  exit 0
fi
# --- 17:35 · E-6 (§12): cuatro sembradas de kc3_s1 (la base de la unica que arranco), otra semilla de datos.
if [ "${1:-}" = "e6" ]; then
  uno_rot nf lat2 3:3 F N
  uno_rot nf lat2 3:4 C L
  uno_rot nf lat2 3:5 M K
  uno_rot nf lat2 3:6 A J
  exit 0
fi
if [ "${1:-}" = "pendientes" ]; then
  uno_rot nd lat2 3:3 N F G J
  uno_rot nd lat2 3:5 C L D I
  uno_rot ne attn 3:1 M K H A
  uno_rot ne attn 3:2 A N E J
  exit 0
fi

uno_rot nd lat2 3:3 N F
uno_rot nd lat2 3:4 G J
uno_rot nd lat2 3:5 C L
uno_rot nd lat2 3:6 D I
uno_rot ne attn 3:1 M K
uno_rot ne attn 3:2 A N
