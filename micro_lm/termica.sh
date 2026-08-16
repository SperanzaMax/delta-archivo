#!/usr/bin/env bash
# Guarda térmica para cualquier trabajo local de CPU.
#
#   Uso:  ./termica.sh            -> imprime la temperatura y sale 0 si es segura, 1 si no
#         ./termica.sh esperar    -> espera hasta que baje del umbral (máx 20 min)
#
# Por qué existe: esta máquina va de ~31 °C en reposo a tocar el crítico de 100 °C en una corrida
# larga de CPU, y el crítico no es una advertencia — es apagado térmico. El umbral de 80 °C que
# reporta `sensors` como "high" es el punto donde el procesador ya está limitando frecuencia, así
# que trabajar por encima de eso además de arriesgado es lento.
#
# REGLA DE FONDO, que es lo que de verdad protege la máquina: el entrenamiento NO se corre acá.
# Medido hoy: 200 pasos en CPU no terminaron en 10 minutos, contra ~0,46 s/paso en una T4 (o sea
# menos de 2 minutos). Correr local no es "la opción de respaldo", es 30 veces más lento y encima
# calienta. La PC guarda checkpoints y orquesta; la GPU entrena.
set -uo pipefail

UMBRAL="${UMBRAL:-75}"
MODO="${1:-ver}"

# Se corta la línea en el paréntesis ANTES de buscar el número. Sin eso se leen los umbrales que
# sensors imprime al lado —"(high = +80.0°C, crit = +100.0°C)"— y el máximo da siempre 100, o sea
# la guarda bloquearía todo para siempre creyendo que la máquina está en el crítico. Detectado al
# probarla: marcó 100 °C con la CPU a 50.
temp() {
  sensors 2>/dev/null \
    | sed 's/(.*//' \
    | grep -oE '\+[0-9]+\.[0-9]+°C' \
    | tr -d '+°C' \
    | sort -rn | head -1 | cut -d. -f1
}

T="$(temp)"
if [ -z "$T" ]; then echo "no se pudo leer la temperatura (¿falta lm-sensors?)"; exit 0; fi

if [ "$MODO" = "esperar" ]; then
  for _ in $(seq 1 40); do
    T="$(temp)"
    [ "$T" -lt "$UMBRAL" ] && { echo "temperatura $T °C — por debajo del umbral $UMBRAL"; exit 0; }
    echo "temperatura $T °C — esperando a que baje de $UMBRAL"
    sleep 30
  done
  echo "sigue en $T °C tras 20 min: no se arranca"; exit 1
fi

echo "temperatura máxima actual: $T °C (umbral $UMBRAL · crítico 100)"
[ "$T" -lt "$UMBRAL" ]
