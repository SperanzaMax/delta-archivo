#!/usr/bin/env bash
# PREREG_RECOMPENSA_L.md (SHA 96e750b6) · el subsidio al silencio.
#
# Ocho unidades sembradas desde b3_s3 y b3_s6 —las dos declaradas atractor absorbente el 29-ago—,
# cruzando L en {0,0 · 0,5} por interfaz en {token · cabeza}. M=0,5 y F=0,2 NO se tocan.
#
# Se lanza POR PRIORIDAD y no todo junto, por dos razones: ayer Colab dio 503 en las trece cuentas
# durante ocho vueltas, y cuatro rotadores sobre el mismo pool se pisan entre si. Las T deciden L-1 y
# L-2, asi que van primero y con el pool partido en dos mitades disjuntas.
#
# OJO con las comillas: la ruta tiene un espacio (`Nuevo Transformer`) y el comando del §6 del
# ESTADO_20260828 salia con exit 127 SIN LANZAR NADA por pasarla pelada a `env`. El .log se creaba
# igual y parecia que habia arrancado.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"

FASE="${1:-T}"          # T = las cuatro de token (principal) · H = las cuatro de cabeza

comun() {
  export P_NOSE=0.4 DONDE=pre BLANCO=error HORIZONTE=12000 SEMBRAR=0 \
         PERDIDA_CABEZA=recompensa REC_M=0.5 REC_F=0.2 REC_CE=1.0
}

lanzar() {   # $1=prefijo  $2=abst  $3=REC_L  $4=log  $5...=cuentas
  local pref="$1" abst="$2" recl="$3" log="$4"; shift 4
  PREFIJO="$pref" ABST="$abst" REC_L="$recl" ACEL=t4 LOG_ROTADOR="$AQUI/$log" \
    setsid ./rotar_abst3.sh 3:3,3:6 3000 3000 250 "$@" \
    > "$AQUI/$log" 2>&1 < /dev/null &
  echo "lanzado $pref (abst=$abst, L=$recl) -> $log  [cuentas: $*]"
}

comun
case "$FASE" in
  T)
    # Pool partido en dos mitades DISJUNTAS para que los dos rotadores no compitan por la misma VM.
    lanzar t0 token 0.0 rot_t0_0830.log A J F D C L K
    sleep 20
    lanzar t5 token 0.5 rot_t5_0830.log H M N I G E
    ;;
  H)
    lanzar h0 cabeza 0.0 rot_h0_0830.log A J F D C L K
    sleep 20
    lanzar h5 cabeza 0.5 rot_h5_0830.log H M N I G E
    ;;
  *)
    echo "uso: $0 [T|H]"; exit 2 ;;
esac
echo
echo "control:  tail -f $AQUI/rot_*_0830.log"
echo "criterio: exactitud global > 0,4065 (L-1) y abstencion entre 0,05 y 0,95 (L-3)"
