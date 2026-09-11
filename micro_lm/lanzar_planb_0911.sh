#!/usr/bin/env bash
# PLAN B AMPLIADO · PREREG_TOPK_ENTRENADO.md §7 · 2026-09-11 08:55
# ts3/ds3 sembrados de kq3, 8.000 pasos, sobre las seis sesiones T4 que quedaron vivas.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=rel PERT=1 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0
       RELLENO=3240 RELLENO_DIST=real RELLENO_TURNOS=viejo RELLENO_CADA=1000 FOTOS=25
       HORIZONTE=8000)
cd "$AQUI"
uno() {  # familia topk unidad cuenta sesion
  env "${COMUN[@]}" PREFIJO="$1" TOPK="$2" setsid nohup ./sesion_fija.sh "$4" "$5" "$3" 8000 2000 500 \
    > "$SAL/fija_${1}3_s${3##*:}.log" 2>&1 < /dev/null &
  echo "$1 $3 en $4/$5 pid $!"; sleep 2
}
if [ "${1:-}" = "primero" ]; then
uno ts 2 3:0 G tr2_g_0830
uno ts 2 3:1 C tr2_c_0829
uno ts 2 3:2 D tr2_d_0828
uno ds 0 3:0 E tr2_e_0830
uno ds 0 3:1 F tr2_f_0828
uno ds 0 3:2 J tr2_j_0828
fi

# --- 09:35 · SEGUNDO INTENTO. El primero murio de OOM en la GPU: los `entrenar.py` de la corrida
# en frio seguian corriendo en las VMs reusadas (matar el tramo en la PC no mata el proceso
# remoto) y JAX preasigna el 75 % de la memoria. Despues, sin keep-alive, las seis sesiones se
# desasignaron solas. Se vuelve a pedir GPU con el rotador normal, un rotador por unidad.
if [ "${1:-}" = "segundo" ]; then
  uno_rot() {  # familia topk unidad cuentas...
    local fam="$1" k="$2" u="$3"; shift 3
    env "${COMUN[@]}" PREFIJO="$fam" TOPK="$k" LOG_ROTADOR="$SAL/rotador_${fam}3_s${u##*:}.log" \
      setsid nohup ./rotar_abst3.sh "$u" 8000 2000 500 "$@" > "$SAL/rotador_${fam}3_s${u##*:}.log" 2>&1 < /dev/null &
    echo "rotador ${fam}3_s${u##*:} pid $! cuentas $*"; sleep 3
  }
  uno_rot ts 2 3:0 H G
  uno_rot ts 2 3:1 K C
  uno_rot ts 2 3:2 L D
  uno_rot ds 0 3:0 M E
  uno_rot ds 0 3:1 N F
  uno_rot ds 0 3:2 I J
fi
