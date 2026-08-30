#!/usr/bin/env bash
# PRUEBA EXPLORATORIA con F=0,2 (ENMIENDA_RECOMPENSA_F). NO juzga W-1 ni W-2, y sus numeros NO se
# usan para volver a elegir pesos: eso seria ajustar sobre la marcha y necesita otro pre-registro.
# Dos unidades de token, que es la condicion principal. 3000 pasos.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"
export P_NOSE=0.4 DONDE=pre BLANCO=error HORIZONTE=26000 SEMBRAR=0 PERDIDA_CABEZA=recompensa
PREFIJO=f2 ABST=token ACEL=t4 LOG_ROTADOR="$AQUI/rot_f2_0829.log" \
  setsid ./rotar_abst3.sh 3:3,3:6 3000 3000 250 A J F D C L K H M N I G E \
  > "$AQUI/rot_f2_0829.log" 2>&1 < /dev/null &
echo "prueba f2 (token, F=0,2, CE=1,0) lanzada"
