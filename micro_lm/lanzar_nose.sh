#!/usr/bin/env bash
# Lanza la campaña de ABSTENCION (p_nose > 0) en todas las cuentas libres.
#
#   Uso:  ./lanzar_nose.sh [cuentas...]      (por defecto, las 13 del pool)
#
# Por qué existe esta campaña aparte: hasta el 2026-08-15 TODAS las corridas usaron p_nose = 0, o
# sea que ninguna pregunta carecía de respuesta en el archivo. Con eso la métrica `nose` sale NaN y
# la abstención ni siquiera es una opción que el modelo pueda tomar: los errores son todos
# silenciosos por construcción. Medido antes de gastar GPU: con p_nose=0.2 las métricas nose,
# nose_ent y nose_rel se computan; con 0.0 las tres son NaN.
#
# Orden de la cola: x1_s0 primero —la COMPUERTA, en el nivel más fácil— y después el nivel 4, que
# es el resultado que importa. Si el modelo no aprende a abstenerse en el nivel fácil, el problema
# es del diseño y conviene enterarse antes de gastar tres corridas de 12 000 pasos.
#
# x1_s1 y x1_s2 quedan FUERA a propósito: para una compuerta alcanza una semilla, y son lo primero
# que se recorta cuando la cuota no da. Se agregan a UNIDADES si sobra presupuesto.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS="$AQUI/logs_campania_$(date +%Y%m%d)"; mkdir -p "$LOGS"
CUENTAS=("${@:-A C D E F G H I J K L M N}")
[ "$#" -eq 0 ] && CUENTAS=(A C D E F G H I J K L M N)

# p_nose = 0.4 y no 0.2, MEDIDO antes de gastar la primera GPU (2026-08-15):
#
#   p_nose   % de preguntas sin respuesta   acierto de NO ABSTENERSE NUNCA
#     0.2              0.2047                          0.7953
#     0.4              0.4094                          0.5906
#
# Con 0.2 la estrategia «contestar siempre algo» vale 0,7953, y la mejor corrida de nivel 4 que
# tenemos saca 0,7598. O sea: el atajo es MEJOR que el modelo entrenado, y el gradiente no tiene
# ningun motivo para aprender a abstenerse. La compuerta habria fallado por diseño nuestro y no por
# incapacidad del modelo — el mismo error que ya se cometió con el control vacio (m=1), con el corte
# prematuro de entrenamiento y con el lr heredado: el sujeto no falla, falla lo que se le da.
#
# A 0.4 el atajo cae a 0,5906 y deja de dominar. La compuerta pregunta «¿PUEDE abstenerse?», no
# «¿lo hace bajo la proporcion final?»: si puede a 0.4 se estudia despues a 0.2, y si no puede a 0.4
# tampoco iba a poder a 0.2.
export PREFIJO=x
export P_NOSE=0.4
export UNIDADES="1:0 4:0 4:1 4:2"

echo "== campaña de abstención · p_nose $P_NOSE · unidades: $UNIDADES"
for c in "${CUENTAS[@]}"; do
  # Nunca dos workers sobre la misma cuenta: son dos procesos `colab` que se pisan el sessions.json
  # y dejan la VM inalcanzable.
  if pgrep -f "worker_cola[.]sh $c " >/dev/null 2>&1; then
    echo "   $c ya tiene worker vivo, se saltea"; continue
  fi
  nohup "$AQUI/worker_cola.sh" "$c" 12000 12000 1000 >> "$LOGS/nose_$c.log" 2>&1 &
  echo "   $c lanzada (pid $!)"
  sleep 3
done
echo "== $(pgrep -c -f "worker_cola[.]sh" || echo 0) workers vivos"
wait
