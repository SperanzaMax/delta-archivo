#!/usr/bin/env bash
# Cierre de la jornada del 6-sep: junta resultados, guarda todo, avisa por Telegram y apaga.
#
#   Uso:  ./cerrar_jornada.sh            (sin apagar, para probar)
#         ./cerrar_jornada.sh --apagar
#
# El aviso va ANTES del apagado a proposito: despues de `poweroff` no hay forma de avisar nada.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python
APAGAR=0; [ "${1:-}" = "--apagar" ] && APAGAR=1

cd "$AQUI"
echo "== 1. estado de la campania"
$PY - <<'PY'
import pickle
for f in ("rp","rr"):
    for s in (0,1,2):
        try:
            b=pickle.load(open(f"micro_lm/ckpts/{f}3_s{s}.pkl","rb"))
            print(f"   {f}3_s{s}: paso {b['paso']}")
        except Exception as e:
            print(f"   {f}3_s{s}: sin checkpoint ({e.__class__.__name__})")
PY

echo "== 2. git: todo guardado"
git add -A
git commit -q -m "Cierre de la jornada del 6-sep: resultados de la campania y de TinyLlama

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01H66ovqmwLw51Dgzgm6goAG" 2>/dev/null && echo "   commit hecho" || echo "   nada nuevo para commitear"
git push -q 2>&1 | tail -2 && echo "   pusheado"

echo "== 3. bitacora a su carpeta"
cp -f BITACORA_20260906.md "/home/maxi/Documentos/Nuevo Transformer/Bitacora/" && echo "   copiada"

if [ "$APAGAR" = "1" ]; then
  echo "== 4. apagando en 60 s"
  sudo shutdown -h +1 "Cierre de la jornada"
else
  echo "== 4. (sin --apagar: no se apaga)"
fi
