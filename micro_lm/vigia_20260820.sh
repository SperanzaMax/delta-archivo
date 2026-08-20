#!/usr/bin/env bash
# Avisa por Telegram el cierre de los DOS experimentos del 20-ago, cada uno con su veredicto.
#
#   Uso:  vigia_20260820.sh <log de la sonda> <log del rotador>
#
# Va como proceso aparte, igual que `avisar_telegram.sh`: si el aviso dependiera de que yo esté
# procesando la sesion, no llegaria cuando Maxi no esta mirando la terminal, que es justo cuando
# sirve. Canal @Albertagente_bot, el unico verificado (los otros dan 403, ver la memoria
# `telegram-notify`).
#
# Las dos esperas corren en paralelo y cada una avisa cuando le toca: el corte sin etiquetas cierra
# en CPU local, el tramo de c4_s2 en una VM de Colab, y no hay razon para que uno espere al otro.
set -uo pipefail

LOG_SONDA="${1:?falta el log de la sonda}"
LOG_ROT="${2:?falta el log del rotador}"

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502
PY=/home/maxi/.venv-ligamento/bin/python
JS="$AQUI/corridas_20260820/c4_s2.json"
LIMITE=720          # 720 x 30 s = 6 h de guardia, de sobra para los dos

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

# ---------------------------------------------------------------- 1 · el corte sin etiquetas
vigilar_sonda() {
  for _ in $(seq 1 $LIMITE); do
    grep -q "^-> " "$LOG_SONDA" 2>/dev/null && break
    # que el proceso haya muerto SIN escribir la salida tambien es un cierre, y hay que avisarlo:
    # el silencio y el exito no se pueden parecer.
    if ! pgrep -f sonda_sin_etiquetas >/dev/null 2>&1; then
      if ! grep -q "^-> " "$LOG_SONDA" 2>/dev/null; then
        mandar "⚠️ micro-LM · corte SIN etiquetas: el proceso murio sin escribir el resultado.
Ultimas lineas:
$(tail -5 "$LOG_SONDA" 2>/dev/null)"
        return
      fi
    fi
    sleep 30
  done
  local s1 s3 s4 s5
  s1="$(grep -E '^S-1 ·' "$LOG_SONDA" | tail -1)"
  s3="$(grep -E '^S-3 ·' "$LOG_SONDA" | tail -1)"
  s4="$(grep -E '^S-4 ·' "$LOG_SONDA" | tail -1)"
  s5="$(grep -E '^S-5 ·' "$LOG_SONDA" | tail -1)"
  mandar "✅ micro-LM · CERRO el corte SIN etiquetas (PREREG SHA 17e0a35e)

$s1
$s4
$s5
$s3

Es la pregunta que separa «la informacion esta en el logit» de «el modelo sabe cuando no sabe»."
}

# ---------------------------------------------------------------- 2 · c4_s2 con mas presupuesto
vigilar_tramo() {
  for _ in $(seq 1 $LIMITE); do
    grep -q '"paso": 20000' "$JS" 2>/dev/null && break
    if grep -qE "sesion perdida" "$LOG_ROT" 2>/dev/null; then
      mandar "⚠️ micro-LM · c4_s2: se cayo la VM a mitad del tramo.
El checkpoint de la PC conserva lo ultimo bajado, se continua desde ahi en otra cuenta."
      return
    fi
    sleep 30
  done
  local paso ver
  paso="$(grep -o '\"paso\": [0-9]*' "$JS" 2>/dev/null | tail -1 | grep -o '[0-9]*')"
  if [ "${paso:-0}" != "20000" ]; then
    mandar "⚠️ micro-LM · c4_s2: se acabo la guardia y el tramo quedo en el paso ${paso:-0} de 20000."
    return
  fi
  # La tendencia sale del JSON y no cuesta nada. T-2 muestrea 2048 sobre dos checkpoints y competiria
  # por la CPU con la sonda, asi que ese lo corro yo aparte.
  ver="$(cd "$AQUI" && taskset -c 3 "$PY" analizar_c4s2.py --sin-extremos 2>&1 | grep -E '^T-1|^T-3|tendencia' | head -4)"
  mandar "✅ micro-LM · CERRO c4_s2 a 20000 pasos (PREREG SHA 8446a27e)

$ver

Falta T-2 (extremos con 2048 muestras), que corre aparte."
}

vigilar_sonda &
vigilar_tramo &
wait
