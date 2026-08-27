#!/usr/bin/env bash
# Mata el tramo que se quedo esperando a una VM que ya no existe.
#
# El 19-ago a las 16:57 `c4_s2` subio el checkpoint a la cuenta K y no imprimio una linea mas
# durante 2h22. La sesion de Colab se habia caido —`colab status` decia «not found»— pero
# `tramo_abst.sh` seguia esperando a un kernel muerto, y el rotador no puede pasar a la cuenta
# siguiente porque espera al tramo. La VM se cae sin aviso, que es la premisa de toda esta infra,
# pero esta fase no tenia timeout.
#
# Regla: si hay un tramo vivo y el log del rotador no crece en LIMITE segundos, el tramo esta
# colgado. Se lo mata y el rotador sigue con la cuenta siguiente por su cuenta.
#
# El umbral es holgado a proposito: durante el entrenamiento el tramo imprime cada 2 min (polling),
# asi que 12 min sin una linea no es lentitud, es que no hay nadie del otro lado.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${1:?falta el log del rotador}"
LIMITE="${LIMITE:-720}"
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo

while pgrep -f rotar_abst.sh >/dev/null; do
  sleep 60
  PID="$(pgrep -f 'tramo_abst.sh' | head -1)"
  [ -n "$PID" ] || continue
  AHORA=$(date +%s)
  MOD=$(stat -c %Y "$LOG" 2>/dev/null || echo "$AHORA")
  QUIETO=$(( AHORA - MOD ))
  if [ "$QUIETO" -ge "$LIMITE" ]; then
    echo "$(date +%H:%M:%S) tramo $PID quieto ${QUIETO}s — se lo mata para destrabar el rotador"
    kill "$PID" 2>/dev/null
    curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
      -d chat_id="$CHAT" --data-urlencode \
      "text=⚠️ micro-LM · watchdog: el tramo estuvo ${QUIETO}s sin escribir (la VM se cayo sin avisar). Matado para que el rotador pase a la cuenta siguiente." >/dev/null 2>&1
    sleep 120
  fi
done
echo "$(date +%H:%M:%S) el rotador termino; watchdog fuera"
