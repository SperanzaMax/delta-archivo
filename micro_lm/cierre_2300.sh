#!/usr/bin/env bash
# CIERRE PROGRAMADO · 2026-09-11 · pedido de Maxi: «vamos a dejar trabajar hasta las 23, después
# guardá todo, subí todo al github y apagá esta pc».
#   1. espera hasta la hora dada;  2. mata los rotadores (no los tramos: primero se baja lo ultimo);
#   3. por cada sesion viva baja ck.pkl y ck_fotos.json y la PARA (si no, la VM sigue quemando);
#   4. empaqueta las peliculas, commit + push;  5. avisa por Telegram;  6. apaga.
#   Uso: cierre_2300.sh [HH:MM] [--sin-apagar]
set -uo pipefail
HORA="${1:-23:00}"; APAGAR=1; [ "${2:-}" = "--sin-apagar" ] && APAGAR=0
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$AQUI/.." && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
PY=/home/maxi/.venv-ligamento/bin/python
. "$AQUI/tg_token.sh" 2>/dev/null || true
mandar(){ [ -n "${TOKEN:-}" ] && curl -s -m 30 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null; true; }
LOG="$AQUI/corridas_$(date +%Y%m%d)/cierre.log"; mkdir -p "$(dirname "$LOG")"
exec >>"$LOG" 2>&1
echo "== cierre programado para las $HORA (apagar=$APAGAR) · $(date)"
while [ "$(date +%H:%M)" \< "$HORA" ]; do sleep 30; done
echo "== $(date) · empieza el cierre"

# 2. rotadores y watchdogs fuera (los tramos siguen, para que no haya dos procesos sobre una cuenta)
for p in $(pgrep -f "rotar_abst3.sh"); do kill "$p" 2>/dev/null; done
for p in $(pgrep -f "watchdog_tramo2.sh"); do kill "$p" 2>/dev/null; done
sleep 2
for p in $(pgrep -f "tramo_abst.sh"); do pkill -TERM -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
sleep 3
for p in $(pgrep -f "venv-colab-cli/bin/colab"); do kill "$p" 2>/dev/null; done
sleep 2
rm -f "$HOME/.colab-pool/en_uso_"*

# 3. por cada cuenta: si hay sesion, bajar lo ultimo y parar
RESUMEN=""
for C in A C D E F G H I J K L M N; do
  if [ "$C" = "A" ]; then CL=( "$COLAB" --auth adc ); else
    [ -f "$HOME/.gcloud-cuenta$C/application_default_credentials.json" ] || continue
    CL=( env CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$C" "$COLAB" --auth adc --config "$HOME/.colab-cuenta$C.json" ); fi
  S="$(timeout 60 "${CL[@]}" sessions 2>&1 | grep -aoE "^\[tr2_[a-z]_[0-9]+\]" | tr -d '[]' | head -1)"
  [ -z "$S" ] && continue
  # que unidad corre ahi: se busca en los logs de los rotadores
  U="$(grep -al "sesion $S" "$AQUI"/corridas_*/rotador_*.log 2>/dev/null | tail -1 | sed -E 's/.*rotador_(.*)\.log/\1/')"
  echo "-- $C / $S -> unidad ${U:-?}"
  if [ -n "$U" ]; then
    T="$(mktemp -d)"
    timeout -k 30 300 "${CL[@]}" download -s "$S" /content/ck.pkl "$T/ck.pkl" >/dev/null 2>&1 && [ -s "$T/ck.pkl" ] && mv "$T/ck.pkl" "$AQUI/ckpts/$U.pkl" && echo "   ckpt bajado"
    timeout -k 30 300 "${CL[@]}" download -s "$S" /content/ck_fotos.json "$T/f.json" >/dev/null 2>&1 && [ -s "$T/f.json" ] && mv "$T/f.json" "$AQUI/ckpts/${U}_fotos.json" && echo "   pelicula bajada"
    timeout -k 30 300 "${CL[@]}" download -s "$S" "/content/salidas/$U.json" "$T/s.json" >/dev/null 2>&1 && [ -s "$T/s.json" ] && mv "$T/s.json" "$AQUI/corridas_$(date +%Y%m%d)/$U.json" && echo "   json bajado"
    rm -rf "$T"
    P="$($PY -c "import pickle,sys;print(pickle.load(open(sys.argv[1],'rb')).get('paso','?'))" "$AQUI/ckpts/$U.pkl" 2>/dev/null)"
    RESUMEN="$RESUMEN$U paso $P · "
  fi
  timeout -k 30 120 "${CL[@]}" stop -s "$S" >/dev/null 2>&1 && echo "   sesion parada"
done

# 4. peliculas empaquetadas (las dos principales de encadenados) y commit
cd "$AQUI"
for u in kc3_s1 kd3_s1; do [ -s "ckpts/${u}_fotos.json" ] && $PY armar_pelicula.py "ckpts/${u}_fotos.json" > "peliculas/pelicula_${u}_20260911.json" 2>/dev/null; done
$PY analizar_topk.py > "corridas_$(date +%Y%m%d)/analisis_cierre.txt" 2>&1 || true
cd "$REPO"
git add micro_lm/corridas_20260911 micro_lm/peliculas micro_lm/controles_20260911 micro_lm/cierre_2300.sh 2>/dev/null
git add -A micro_lm/*.py micro_lm/*.sh micro_lm/*.md 2>/dev/null
git commit -q -m "CIERRE 23:00 · estado al apagar: $RESUMEN

Rotadores y sesiones de Colab parados; ultimos checkpoints, peliculas y JSON bajados.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01V6cGBFjBLATxXKEr5vTZ95" && git push -q origin main && echo "== pusheado $(git log --oneline -1)"

# 5. aviso
mandar "11-sep 23:00 · CIERRE AUTOMÁTICO
Parado todo y subido a GitHub ($(git log --oneline -1 | cut -c1-7)).
Estado al apagar: ${RESUMEN:-sin sesiones vivas}
Los JSON con las evaluaciones están en corridas_20260911/; mañana los leo y actualizo el tablero.
$( [ $APAGAR = 1 ] && echo 'Apagando la PC.' || echo '(sin apagar: prueba)')"

# 6. apagar
if [ "$APAGAR" = 1 ]; then
  sleep 10
  systemctl poweroff || sudo -n systemctl poweroff || mandar "⚠ No pude apagar la PC (sin permiso). Todo lo demás está guardado."
fi
