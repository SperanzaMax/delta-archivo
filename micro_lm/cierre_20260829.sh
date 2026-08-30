#!/usr/bin/env bash
# Cierre ordenado del 29-ago, antes de apagar la PC.
#
# Hace lo mismo que `cierre_20260828.sh` y por las mismas razones, mas una que aparecio hoy: los
# `tramo_abst.sh` SOBREVIVEN reparentados si se mata solo al rotador, y siguen bajando checkpoints
# pisando lo que escriba una corrida posterior (el 28 le costo a `b3_s3` un retroceso de 20000 a
# 19000). Por eso se barren los hijos explicitamente.
#
# Y para al final las sesiones de Colab que hayan quedado vivas, porque una VM abierta sigue gastando
# cuota de una cuenta prestada aunque la PC este apagada.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab

echo "== 1. rotadores"
pkill -f "rotar_abst3.sh" 2>/dev/null && echo "   rotadores parados" || echo "   no habia"
sleep 2

echo "== 2. tramos hijos (los que sobreviven reparentados)"
pkill -f "tramo_abst.sh" 2>/dev/null && echo "   tramos barridos" || echo "   no habia"
sleep 2

echo "== 3. vigias y archivadores"
pkill -f "vigia_perdida.sh" 2>/dev/null && echo "   vigia parado" || echo "   no habia"
pkill -f "archivar_traza.sh" 2>/dev/null && echo "   archivador parado" || echo "   no habia"

echo "== 4. sesiones de Colab que hayan quedado abiertas hoy"
SES="$(grep -ho "tr2_[a-z]_[0-9]*" "$AQUI"/rot_*_0829.log 2>/dev/null | sort -u)"
if [ -z "$SES" ]; then
  echo "   no se encontraron sesiones en los logs de hoy"
else
  for s in $SES; do
    c="$(echo "$s" | cut -d_ -f2 | tr 'a-z' 'A-Z')"
    if [ "$c" = "A" ]; then
      unset CLOUDSDK_CONFIG
      r="$(timeout 60 "$COLAB" --auth adc stop -s "$s" 2>&1 | tail -1)"
    else
      r="$(CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$c" timeout 60 "$COLAB" --auth adc \
           --config "$HOME/.colab-cuenta$c.json" stop -s "$s" 2>&1 | tail -1)"
    fi
    echo "   $s -> $r"
  done
fi

echo "== 5. locks del pool"
rm -f "$HOME/.colab-pool"/en_uso_* 2>/dev/null && echo "   locks limpios" || echo "   no habia"

echo
echo "== verificacion final"
n=$(ps -eo args --no-headers | grep -cE "^(bash )?\./(rotar_abst3|tramo_abst|vigia_perdida|archivar_traza)\.sh" || true)
echo "   procesos del proyecto vivos: $n"
ls "$HOME/.colab-pool"/en_uso_* >/dev/null 2>&1 && echo "   OJO: quedan locks" || echo "   sin locks"
command -v sensors >/dev/null && sensors 2>/dev/null | grep Package
echo
[ "$n" = "0" ] && echo "CIERRE OK · se puede apagar" || echo "CIERRE INCOMPLETO · revisar antes de apagar"
