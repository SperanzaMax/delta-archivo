#!/usr/bin/env bash
# Cierre del 31-ago. Para todo, guarda y deja la PC libre. NO apaga salvo que se pase --apagar:
# Maxi pidio «detener y guardar», no apagar, asi que el apagado va detras de un flag explicito.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"; REPO="$(cd .. && pwd)"
CB=/home/maxi/.venv-colab-cli/bin/colab
APAGAR=0; [ "${1:-}" = "--apagar" ] && APAGAR=1

echo "== 1. parando rotadores, tramos y mediciones de CPU"
pkill -f rotar_abst3 2>/dev/null; pkill -f tramo_abst 2>/dev/null
pkill -f sonda_techo 2>/dev/null; pkill -f desacuerdo_busq 2>/dev/null; pkill -f juzgar_ 2>/dev/null
sleep 3
pkill -9 -f rotar_abst3 2>/dev/null; pkill -9 -f tramo_abst 2>/dev/null
echo "   procesos vivos: $(pgrep -f 'rotar_abst3|tramo_abst|sonda_techo|desacuerdo_busq' | wc -l)"

echo "== 2. parando sesiones de Colab en las 13 cuentas"
for C in A J F D C L K H M N I G E; do
  if [ "$C" = "A" ]; then CFG=(); else CFG=(--config "$HOME/.colab-cuenta$C.json"); fi
  SES=$(timeout 40 "$CB" --auth adc "${CFG[@]}" sessions 2>/dev/null | grep -o 'tr2_[a-z0-9_]*' | sort -u)
  for S in $SES; do
    echo "   parando $S en cuenta $C"
    timeout 60 "$CB" --auth adc "${CFG[@]}" stop -s "$S" >/dev/null 2>&1
  done
done

echo "== 3. guardando en git"
cd "$REPO"; git add -A
git commit -q -m "Cierre del 31-ago: el slot con orden colapsa, y falta el brazo del peso chico

Queda corriendo o listo para juzgar el brazo W-8 (v03_s3 y v03_s6, --rec-rank 0,008), que es el que
decide si el colapso lo causo la interfaz o el peso que derive. El plan de mañana esta en
PLAN_20260901.md, con el orden y los criterios ya escritos.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016j6MVbYbN7paNMUQjNVZRf" || echo "   (nada nuevo que commitear)"
git push -q origin HEAD && echo "   pusheado: $(git rev-parse HEAD | cut -c1-8)" || echo "   ** push FALLO, el trabajo queda solo local **"
echo "   sin commitear: $(git status --porcelain | wc -l) archivos"

echo "== 4. verificacion final"
echo "   checkpoints de hoy: $(ls -1 "$AQUI"/ckpts/{k03,w03,v03}_s*.pkl 2>/dev/null | wc -l) de 6"
sensors 2>/dev/null | grep "Package id 0"

if [ "$APAGAR" = "1" ]; then echo "== 5. APAGANDO"; sleep 5; sudo systemctl poweroff
else echo "== listo. NO se apaga (pasar --apagar si se quiere)"; fi
