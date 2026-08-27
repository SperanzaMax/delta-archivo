#!/usr/bin/env bash
# Avisa por Telegram como viene A5 cada 15 min, y manda un resumen al terminar.
#
# Pedido de Maxi el 27-ago. Va como proceso aparte y no como un loop de la sesion a proposito: si el
# aviso dependiera de que yo este procesando la conversacion, no llegaria justo cuando el no esta
# mirando la terminal, que es cuando sirve.
#
# Termina solo en dos casos, y los dos importan:
#   · las tres unidades llegan a 26000  -> resumen final
#   · no queda rotador vivo             -> aviso de caida
# Sin la segunda, un rotador muerto se veria igual que «sigue corriendo».
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$AQUI/tg_token.sh"
PY=/home/maxi/.venv-ligamento/bin/python
CADA="${CADA:-900}"
TOPE=26000

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

paso_de() {
  "$PY" -c "
import pickle,sys
try:
    print(pickle.load(open('$AQUI/ckpts/$1.pkl','rb')).get('paso') or 0)
except Exception:
    print(0)" 2>/dev/null
}

# Ultima eval de tramo de una unidad, buscada en los logs de hoy. Es de 512 muestras: sirve para la
# TENDENCIA, no para el nivel, y el aviso lo dice cada vez para que nadie lea un 1,0000 como final.
eval_de() {
  grep -h "ULTIMO= ── eval" "$AQUI"/rot_b3*_$(date +%Y%m%d).log 2>/dev/null | tail -1 \
    | sed 's/.*ULTIMO= ── eval: //' | cut -c1-120
}

VUELTA=0
while true; do
  P0=$(paso_de b3_s0); P1=$(paso_de b3_s1); P2=$(paso_de b3_s2)
  VIVOS=$(pgrep -f "rotar_abst3.sh" | wc -l)
  LISTAS=0
  for p in "$P0" "$P1" "$P2"; do [ "${p:-0}" -ge "$TOPE" ] && LISTAS=$((LISTAS+1)); done

  if [ "$LISTAS" -ge 3 ]; then
    mandar "🏁 A5 TERMINÓ. Las tres unidades en $TOPE.

b3_s0 $P0 · b3_s1 $P1 · b3_s2 $P2

Las cuentas de Colab quedan libres. Ahora corro el SER a cobertura igualada en las tres contra p3_* y te paso el veredicto de E-1, que se decide por 2 de 3.

Ojo: b3_s0 ya dio -0,0137 a cobertura 0,70, y E-1 pedía 0,02. O sea que s0 NO cumple, y hacen falta las otras dos."
    exit 0
  fi

  if [ "$VIVOS" -eq 0 ]; then
    mandar "🔴 A5: NO QUEDA NINGÚN ROTADOR VIVO y sólo $LISTAS de 3 unidades están completas.

b3_s0 $P0 · b3_s1 $P1 · b3_s2 $P2

Los checkpoints de la PC conservan lo bajado, así que no se pierde progreso, pero hay que relanzar."
    exit 0
  fi

  VUELTA=$((VUELTA+1))
  mandar "⏱️ A5 · vuelta $VUELTA · $(date +%H:%M)

b3_s0  $P0/$TOPE
b3_s1  $P1/$TOPE
b3_s2  $P2/$TOPE
(faltan $(( (TOPE-P0) + (TOPE-P1) + (TOPE-P2) )) pasos en total)

última eval de tramo: $(eval_de)
Es de 512 muestras, sirve para la tendencia y no para el nivel.

rotadores vivos: $VIVOS"

  sleep "$CADA"
done
