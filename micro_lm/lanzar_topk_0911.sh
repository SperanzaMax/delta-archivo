#!/usr/bin/env bash
# LA CAMPANIA DEL TOP-K ENTRENADO · PREREG_TOPK_ENTRENADO.md (03b294f9) · 2026-09-11
#
#   tp3_s{0,1,2}  top-2 desde el paso 0 + relleno 3240 real viejo    (la principal)
#   dp3_s{0,1,2}  softmax completo + el MISMO relleno                (el control)
#
# Dos rotadores con listas de cuentas DISJUNTAS: el lock por cuenta ya evita que se pisen, pero
# con listas separadas ninguno espera al otro. La cuenta A queda para el piloto del corpus.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAL="$AQUI/corridas_$(date +%Y%m%d)"; mkdir -p "$SAL"
COMUN=(SELLO=rel PERT=1 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.4 ABST=cabeza SEMBRAR=0
       RELLENO=3240 RELLENO_DIST=real RELLENO_TURNOS=viejo RELLENO_CADA=1000 FOTOS=100
       HORIZONTE=26000 VUELTAS=12)
cd "$AQUI"
if [ "${1:-}" = "primera" ]; then
env "${COMUN[@]}" PREFIJO=tp TOPK=2 LOG_ROTADOR="$SAL/rotador_tp.log" \
  setsid nohup ./rotar_abst3.sh 3:0,3:1,3:2 26000 2000 1000 H K L G C D > "$SAL/rotador_tp.log" 2>&1 < /dev/null &
echo "rotador tp pid $!"
sleep 5
env "${COMUN[@]}" PREFIJO=dp TOPK=0 LOG_ROTADOR="$SAL/rotador_dp.log" \
  setsid nohup ./rotar_abst3.sh 3:0,3:1,3:2 26000 2000 1000 M N I E F J > "$SAL/rotador_dp.log" 2>&1 < /dev/null &
echo "rotador dp pid $!"
fi

# --- SEGUNDA TANDA, misma manana. El rotador es SECUENCIAL (una cuenta, y en ella los tramos de
# todas las unidades pendientes en serie), asi que con seis unidades de 26.000 pasos tardaria un
# dia. Se lanza UN rotador por unidad restante, y cada uno deja su pid en `ckpts/<u>.local.lock`,
# que es el lock por unidad que el rotador ya respeta (`bloqueada`): los dos rotadores de la
# primera tanda ven s1 y s2 tomadas y no las tocan.
lanzar_uno() {   # familia topk unidad cuentas...
  local fam="$1" k="$2" u="$3"; shift 3
  env "${COMUN[@]}" PREFIJO="$fam" TOPK="$k" LOG_ROTADOR="$SAL/rotador_${fam}_${u##*:}.log" \
    setsid nohup ./rotar_abst3.sh "$u" 26000 2000 1000 "$@" > "$SAL/rotador_${fam}_${u##*:}.log" 2>&1 < /dev/null &
  local pid=$!
  echo "$pid" > "$AQUI/ckpts/${fam}3_s${u##*:}.local.lock"
  echo "rotador ${fam}3_s${u##*:} pid $pid (lock escrito) cuentas $*"
  sleep 3
}
if [ "${1:-}" = "segunda" ]; then
  lanzar_uno tp 2 3:1 K C
  lanzar_uno tp 2 3:2 L D
  lanzar_uno dp 0 3:1 N F
  lanzar_uno dp 0 3:2 I J
fi

# --- TERCERA, misma manana (09:20). Los dos rotadores de la primera tanda se MATARON: son
# secuenciales y, terminado el tramo de s0, habrian seguido con s1 y s2 en la misma sesion,
# pisando los checkpoints de los rotadores de la segunda tanda (el lock `.local.lock` no sirve
# para esto: el rotador borra el propio y la lista FALTAN se calcula antes del tramo). Costo: ~20
# min de T4 en tp3_s0 y dp3_s0, sin checkpoint parcial todavia. Se relanzan como rotadores de UNA
# unidad, que es la forma que no se pisa.
if [ "${1:-}" = "tercera" ]; then
  lanzar_uno tp 2 3:0 H G
  lanzar_uno dp 0 3:0 M E
fi
