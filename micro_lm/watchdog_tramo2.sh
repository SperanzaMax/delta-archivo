#!/usr/bin/env bash
# Igual que `watchdog_tramo.sh`, pero ATADO A UN ROTADOR. Uso: watchdog_tramo2.sh <log> <pid_rotador>
#
# Por que hizo falta (2026-08-20 noche, primera vez que corren DOS campañas a la vez — `token` y
# `escala`): el v1 elige la victima con `pgrep -f 'tramo_abst.sh' | head -1`, o sea EL PRIMERO que
# encuentre, y vigila un solo log. Con dos rotadores vivos eso falla de las dos maneras posibles:
#
#   · el log que vigila crece (campaña sana) mientras la OTRA campaña se cuelga -> nunca actua, y la
#     colgada se traba sin aviso, que es exactamente lo que costo 3h47 el 19-ago;
#   · el log que vigila se queda quieto y el «primer» tramo resulta ser el de la otra campaña -> mata
#     un tramo SANO.
#
# El arreglo es identificar al tramo por PARENTESCO en vez de por nombre: el rotador lanza su tramo
# como hijo directo, asi que `pgrep -P $ROT` no puede confundirse de campaña. Y el corte del bucle
# pasa a ser ESE rotador y no «algun rotar_abst.sh vivo», que con dos campañas tampoco distinguia.
set -uo pipefail
LOG="${1:?falta el log del rotador}"
ROT="${2:?falta el pid del rotador}"
# LIMITE (2026-08-24): pasa de 720 a 5400 segundos, y no es un numero a ojo.
#
# Con 720 el watchdog mataba TRAMOS SANOS de forma sistematica, y la mañana del 24-ago lo hizo con
# el pool por fin abierto despues de cuatro dias secos: v3_s1 arranco en la cuenta H a las 08:18 y a
# las 08:29 ya estaba pidiendo otra cuenta, once minutos despues. Los mensajes decian "el tramo
# estuvo 4063s sin escribir", que era cierto y no era un cuelgue.
#
# La causa es estructural, no un ajuste fino: el tramo corre por `colab exec`, que NO devuelve
# salida hasta terminar, asi que entre la linea "== tramo ..." y el resultado pasan 40-70 minutos en
# los que el log no crece aunque la GPU este entrenando a pleno. El watchdog media exactamente eso y
# lo llamaba cuelgue.
#
# 5400 s (90 min) queda por encima de un tramo completo de 8000 pasos y muy por debajo del episodio
# que motivo este watchdog (3h47 el 19-ago), asi que sigue cubriendo el caso para el que se escribio.
LIMITE="${LIMITE:-5400}"
TOKEN="8723956710:AAE_v0u5y3hDVWePCtKCuGnuY2yDCkRHicw"
CHAT=7985522502

echo "$(date +%H:%M:%S) watchdog2 sobre el rotador $ROT · log $(basename "$LOG") · limite ${LIMITE}s"
while kill -0 "$ROT" 2>/dev/null; do
  sleep 60
  PID="$(pgrep -P "$ROT" -f 'tramo_abst.sh' | head -1)"
  [ -n "$PID" ] || continue
  AHORA=$(date +%s)
  MOD=$(stat -c %Y "$LOG" 2>/dev/null || echo "$AHORA")
  QUIETO=$(( AHORA - MOD ))
  if [ "$QUIETO" -ge "$LIMITE" ]; then
    echo "$(date +%H:%M:%S) tramo $PID (rotador $ROT) quieto ${QUIETO}s — se lo mata"
    kill "$PID" 2>/dev/null
    curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
      -d chat_id="$CHAT" --data-urlencode \
      "text=⚠️ micro-LM · watchdog2 (rotador $ROT): el tramo estuvo ${QUIETO}s sin escribir. Matado para que el rotador pase a la cuenta siguiente." >/dev/null 2>&1
    sleep 120
  fi
done
echo "$(date +%H:%M:%S) el rotador $ROT termino; watchdog2 fuera"
