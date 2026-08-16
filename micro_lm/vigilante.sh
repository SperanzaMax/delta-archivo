#!/usr/bin/env bash
# Vigilante de la campaña (2026-08-15). Corre en segundo plano y TERMINA cuando hay algo que mirar.
#
# Por qué termina en vez de avisar y seguir: quien lo lanza se entera por la notificación de fin de
# proceso, así que salir ES el aviso. Un vigilante que sigue corriendo después de detectar el
# problema no le avisa a nadie.
#
# Lo que busca, en orden de daño:
#  1. DUPLICADO: dos tramos sobre la misma unidad -> dos procesos escribiendo el mismo checkpoint.
#     Es lo que pasó a la mañana con n4_s0 y es la única falla que destruye trabajo ya hecho.
#  2. VM HUERFANA: una sesión viva en una cuenta que no tiene tramo corriendo -> compute units
#     quemándose de gratis hasta el tope de 24 h.
#  3. SEQUIA: nadie consigue acelerador en 30 min -> se acabó la disponibilidad del día (ayer pasó
#     a la tarde), no tiene sentido seguir esperando sin avisar.
#  4. FIN: las 12 unidades llegaron al objetivo.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python
COLAB=/home/maxi/.venv-colab-cli/bin/colab
PASOS="${1:-12000}"
CUENTAS="${2:-A C D E F G H I J}"
LATIDO="$AQUI/logs_campania_$(date +%Y%m%d)/vigilante.log"
sin_gpu=0

paso_de() {
  local f="$AQUI/ckpts/$1.pkl"
  [ -f "$f" ] || { echo 0; return; }
  "$PY" -c "import pickle;print(pickle.load(open('$f','rb'))['paso'])" 2>/dev/null || echo 0
}

while true; do
  # --- 1. unidades duplicadas entre tramos vivos
  # una linea por tramo, con la cuenta y la unidad: "F 4:0". Se saca por patron y no por numero de
  # campo porque pgrep antepone el interprete de forma despareja y el $5 termina siendo la cuenta.
  tramos="$(pgrep -a -f "tramo_colab[.]sh" \
            | grep -oE "tramo_colab\.sh [A-Z]+ [^ ]+ [0-9]+:[0-9]+" \
            | awk '{print $2" "$4}' | sort -u)"
  vivos="$(echo "$tramos" | grep -c ':' || true)"
  dup="$(echo "$tramos" | awk '{print $2}' | sort | uniq -d)"
  if [ -n "$dup" ]; then
    echo "🚨 DUPLICADO: dos tramos sobre la misma unidad -> $dup"
    pgrep -a -f "tramo_colab[.]sh" | sed 's/.*tramo_colab/tramo_colab/'
    exit 2
  fi

  # --- 2. VMs vivas sin nadie que las use
  #
  # Sólo se consultan las cuentas que no tienen NINGUN proceso propio (ni worker ni tramo). La regla
  # es dura y ya costó dos VMs: dos comandos `colab` a la vez sobre la misma cuenta se pisan el
  # sessions.json y lo dejan vacío, y con el registro perdido la sesión queda inalcanzable por CLI —
  # el token JWT vive sólo ahí. Un worker vivo puede estar en medio de un `new`, así que preguntarle
  # a su cuenta es exactamente lo que no hay que hacer. Y no hace falta: mientras el worker viva, él
  # apaga su VM. La huérfana sólo aparece cuando no queda nadie a cargo, que es este caso.
  for c in $CUENTAS; do
    # el script de cierre de F también es "alguien a cargo": apaga esa VM en menos de un minuto
    ocupada() {
      pgrep -f "worker_cola[.]sh $c " >/dev/null 2>&1 && return 0
      pgrep -f "tramo_colab[.]sh $c " >/dev/null 2>&1 && return 0
      [ "$c" = "F" ] && pgrep -f "cerrar_f_y_relanzar" >/dev/null 2>&1 && return 0
      return 1
    }
    ocupada && continue
    if [ "$c" = "A" ]; then CL=("$COLAB" --auth adc)
    else CL=(env "CLOUDSDK_CONFIG=$HOME/.gcloud-cuenta$c" "$COLAB" --auth adc --config "$HOME/.colab-cuenta$c.json"); fi
    salida="$(timeout 90 "${CL[@]}" sessions 2>&1)"
    if ! echo "$salida" | grep -q "No active sessions"; then
      # puede ser un `new` recién lanzado por un worker que arranca: se confirma un minuto después
      sleep 60
      ocupada && continue
      salida2="$(timeout 90 "${CL[@]}" sessions 2>&1)"
      if ! echo "$salida2" | grep -q "No active sessions"; then
        echo "🚨 VM HUERFANA en la cuenta $c (quemando unidades sin trabajo):"
        echo "$salida2" | tail -6
        exit 3
      fi
    fi
  done

  # --- 3. sequía de aceleradores
  if [ "$vivos" -eq 0 ]; then sin_gpu=$((sin_gpu + 1)); else sin_gpu=0; fi
  if [ "$sin_gpu" -ge 15 ]; then
    echo "⚠️  SEQUIA: 30 min sin que ninguna cuenta consiga acelerador."
    exit 4
  fi

  # --- 4. ¿terminó todo?
  falta=0; resumen=""
  for n in 1 2 3 4; do for s in 0 1 2; do
    p="$(paso_de "n${n}_s${s}")"
    [ "$p" -ge "$PASOS" ] || falta=$((falta + 1))
    resumen="$resumen n${n}_s${s}=$p"
  done; done
  echo "$(date +%H:%M) tramos=$vivos faltan=$falta ·$resumen" >> "$LATIDO"
  if [ "$falta" -eq 0 ]; then
    echo "✅ CAMPAÑA COMPLETA: las 12 unidades llegaron a $PASOS pasos."
    exit 0
  fi

  sleep 120
done
