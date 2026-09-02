#!/usr/bin/env bash
# Le manda a Maxi el estado de TODO cada rato, y avisa fuerte si algo se cae.
# Va como proceso aparte a proposito: el reporte no puede depender de que yo este mirando.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
. ./tg_token.sh
CADA="${CADA:-2700}"          # 45 min
mandar(){ curl -s -m 30 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1; }

esperados=3                    # rotadores que deberian estar vivos
for i in $(seq 1 40); do       # ~30 h de cobertura
  sleep "$CADA"
  cuerpo="$(./estado_todo.sh 2>&1)"
  vivos=$(ps aux | grep -c "[r]otar_abst3")
  alerta=""
  # cada rotador arranca 2 procesos (el wrapper y el script), de ahi el x2
  [ "$vivos" -lt $((esperados)) ] && alerta="
⚠️ OJO, hay menos rotadores vivos de los que deberia ($vivos). Alguna campaña se cayó."
  # una campania que no avanza en 45 min con el rotador vivo tambien es sospechosa
  mandar "📊 avance $(date +%H:%M)

$cuerpo$alerta"
done
