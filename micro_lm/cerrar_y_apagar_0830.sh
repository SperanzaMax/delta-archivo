#!/usr/bin/env bash
# Cierre del 30-ago: para todo, guarda, pushea y APAGA. Pedido explicito de Maxi (antes de 23:10).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
AQUI="$PWD"
REPO="$(cd .. && pwd)"
CB=/home/maxi/.venv-colab-cli/bin/colab

echo "== 1. parando rotadores y tramos"
pkill -f rotar_abst3 2>/dev/null; pkill -f tramo_abst 2>/dev/null
sleep 3
pkill -9 -f rotar_abst3 2>/dev/null; pkill -9 -f tramo_abst 2>/dev/null
echo "   procesos vivos: $(pgrep -f 'rotar_abst3|tramo_abst' | wc -l)"

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
cd "$REPO"
git add -A
git commit -q -F - <<'MSG' || echo "   (nada nuevo que commitear)"
Cierre del 30-ago: fase H termina muda, y la lectura estaba comprometida antes

Las cuatro unidades de la interfaz cabeza llegaron a 3000 pasos con abstencion
1,0000 exacta. Por definicion eso da exactitud 0,4065, el piso trivial, y el §4
del prereg ya lo declaraba sin necesidad de medirlo.

NO se lee como fracaso de la interfaz. La NOTA_LECTURA_FASE_H (4a0900bf),
congelada con la campaña corriendo y ANTES del dato, dejo escrito que en esa
celda el desenlace es indistinguible de falta de presupuesto: token arranco
locuaz y cabeza arranca muda, con 3000 pasos para las dos, y el aviso del 26-ago
dice que unidades asi se abstienen del 100% durante ~3000 pasos y despues
aflojan solas.

Entonces L-1 en cabeza queda NO EVALUABLE, y con ella L-4. El criterio de
abandono del §7 NO se aplica: exige las dos interfaces y una no fue medida en
condiciones comparables. Contarlo como negativo seria el quinto negativo por
impaciencia del proyecto.

Lo que si queda establecido: en token, q es una CONSTANTE (~0,50 en las cuatro,
con falsa_abst ~0,48) robusta a semilla, origen y L. L era efectivamente un
subsidio al silencio (+0,0845 de recompensa neta al mudo) pero quitarlo no
alcanza: saca del extremo y deposita en otra constante.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_0172fGRcnejEQwK1rXaQ1Pk5
MSG
git push -q origin HEAD && echo "   pusheado: $(git rev-parse HEAD | cut -c1-8) == origin $(git rev-parse origin/main | cut -c1-8)"
echo "   sin commitear: $(git status --porcelain | wc -l) archivos"

echo "== 4. verificacion final"
echo "   procesos: $(pgrep -f 'rotar|tramo_|entrenar.py|juzgar_L' | wc -l)"
sensors 2>/dev/null | grep "Package id 0"

echo "== 5. APAGANDO"
sleep 5
sudo systemctl poweroff
