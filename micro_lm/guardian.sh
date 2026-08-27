#!/usr/bin/env bash
# Guardian de la campania del 17-ago: mantiene vivo lo que tiene que estar vivo y avisa por Telegram
# cuando pasa algo que merece la atencion de Maxi. Corre solo, sin sesion de agente detras.
#
#   Uso:  guardian.sh [unidades_colab] [pasos]
#   Ej.:  guardian.sh 3:1,4:2 12000
#
# Se diferencia de `vigilante.sh` (15-ago) en una cosa que importa: aquel consulta el CLI de Colab
# para buscar VMs huerfanas, y eso significa un segundo proceso `colab` sobre una cuenta que el
# rotador puede estar usando — justo lo que deja el sessions.json vacio y la VM inalcanzable. Este
# mira SOLO disco y tabla de procesos, asi que no puede interferir con nada.
#
# Tampoco termina al detectar algo: repara lo reparable (relanzar un rotador o un reporte caido) y
# sigue. Terminar era razonable cuando quien lo lanzaba miraba la notificacion; aca el aviso va por
# Telegram, asi que puede quedarse.
set -uo pipefail

UNIDADES="${1:-3:1,4:2}"
PASOS="${2:-12000}"
CADA_VUELTA="${CADA_VUELTA:-300}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
LOGS="$AQUI/logs_campania_$(date +%Y%m%d)"; mkdir -p "$LOGS"
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo
CUENTAS_ROT="H K L M N I G C D E F J A"

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}
log() { echo "[$(date +%H:%M:%S)] $*"; }

# Marcas para no repetir el mismo aviso en cada vuelta. Un guardian que avisa lo mismo cada 5 min
# entrena a Maxi a ignorarlo, que es peor que no avisar.
avisado() { [ -f "$LOGS/.aviso_$1" ]; }
marcar()  { touch "$LOGS/.aviso_$1"; }

completa() {
  local n="${1%%:*}"
  local s="${1##*:}"
  grep -q "\"paso\": $PASOS" "$SALIDA/n${n}_s${s}.json" 2>/dev/null
}

todas_completas() {
  for u in $(echo "$UNIDADES" | tr ',' ' '); do completa "$u" || return 1; done
  return 0
}

log "guardian arriba · unidades $UNIDADES · pasos $PASOS"
while true; do
  # ---- 1. el rotador de Colab -------------------------------------------------------------------
  if ! pgrep -f "rotar[_]tramos.sh" >/dev/null 2>&1; then
    if todas_completas; then
      if ! avisado fin_base; then
        marcar fin_base
        mandar "✅ micro-LM · la campaña base está COMPLETA: las 12 unidades llegaron al paso $PASOS."
        log "campania base completa"
      fi
    else
      # Murio con trabajo pendiente: relanzarlo es siempre correcto porque el rotador reanuda desde
      # el checkpoint y saltea lo ya terminado.
      log "el rotador no esta vivo y quedan pendientes: relanzando"
      mandar "⚠️ micro-LM · el rotador de Colab se había caído con unidades pendientes. Lo relancé; retoma desde el último checkpoint."
      # shellcheck disable=SC2086
      setsid nohup "$AQUI/rotar_tramos.sh" "$UNIDADES" "$PASOS" 8000 500 $CUENTAS_ROT \
        >> "$LOGS/rotador.log" 2>&1 < /dev/null &
      sleep 30
    fi
  fi

  # ---- 2. el reporte de cada 30 min -------------------------------------------------------------
  if ! pgrep -f "reporte30.sh" >/dev/null 2>&1; then
    log "el reporte de 30 min no esta vivo: relanzando"
    setsid nohup "$AQUI/reporte30.sh" "$UNIDADES" "$PASOS" 30 \
      >> "$LOGS/reporte30.log" 2>&1 < /dev/null &
    sleep 5
  fi

  # ---- 3. la prueba local del curriculum de abstencion ------------------------------------------
  XJS="$SALIDA/x1_s0.json"
  if [ -f "$XJS" ]; then
    FA="$(grep -o '"falsa_abst": [0-9.]*' "$XJS" | tail -1 | cut -d' ' -f2)"
    VIG="$(grep -o '"vigente": [0-9.]*' "$XJS" | tail -1 | cut -d' ' -f2 | cut -c1-6)"
    NOSE="$(grep -o '"nose": [0-9.]*' "$XJS" | tail -1 | cut -d' ' -f2 | cut -c1-6)"
    PASO_X="$(grep -o '"paso": [0-9]*' "$XJS" | tail -1 | grep -o '[0-9]*')"
    # El colapso es el resultado que ya se vio el 15-ago arrancando de un modelo sin saturar. Si
    # vuelve a pasar ARRANCANDO DE 1,0000, la conclusion es mucho mas fuerte —y no hace falta
    # esperar los 1000 pasos para saberlo—.
    if [ -n "$FA" ] && awk "BEGIN{exit !($FA >= 0.90)}" && ! avisado colapso_x1; then
      marcar colapso_x1
      mandar "🔬 micro-LM · x1_s0 (currículum de abstención) COLAPSÓ en el paso $PASO_X: falsa_abst $FA · vigente $VIG.
Arrancó de un modelo saturado en 1,0000, así que esta vez el colapso NO se explica por «el mecanismo todavía no rinde»."
    fi
    if [ -n "$FA" ] && awk "BEGIN{exit !($FA <= 0.10)}" && [ -n "$NOSE" ] \
       && awk "BEGIN{exit !($NOSE >= 0.50)}" && ! avisado exito_x1; then
      marcar exito_x1
      mandar "🎉 micro-LM · x1_s0 PASA LA COMPUERTA en el paso $PASO_X: nose $NOSE · falsa_abst $FA · vigente $VIG.
El currículum funciona: el modelo aprende a abstenerse sin abstenerse de todo."
    fi
  fi

  # ---- 4. temperatura ---------------------------------------------------------------------------
  T="$(sensors 2>/dev/null | sed 's/(.*//' | grep -oE '\+[0-9]+\.[0-9]+°C' | tr -d '+°C' \
        | sort -rn | head -1 | cut -d. -f1)"
  if [ -n "$T" ] && [ "$T" -ge 90 ] && ! avisado temp_alta; then
    marcar temp_alta
    mandar "🌡 micro-LM · la CPU llegó a $T °C. La guarda de local_tramo.sh pausa sola en 82, así que esto indica que hay OTRA carga pesada en la máquina."
  fi

  sleep "$CADA_VUELTA"
done
