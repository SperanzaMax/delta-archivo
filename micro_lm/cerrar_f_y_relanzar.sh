#!/usr/bin/env bash
# Cierre puntual del 2026-08-15: el tramo de F quedó HUERFANO a propósito.
#
# Contexto: la primera tanda del día arrancó con la versión vieja de worker_cola.sh, que reclamaba
# la unidad DESPUES de conseguir la VM y por eso cuatro cuentas tomaron n4_s0 a la vez. Al parchear
# el worker no se podía reiniciar el de F sin tirar un tramo que ya iba por el paso 8000, así que se
# mató al padre y se dejó vivo al hijo. Este script hace lo que habría hecho el padre: esperar el
# tramo, apagar la VM, soltar el reclamo y devolver F a la cola con el worker parcheado.
#
# La primera versión relanzaba también a D. Se sacó: D volvió antes por su cuenta, y dejarlo acá
# habría puesto DOS workers sobre la misma cuenta —dos procesos `colab` que se pisan el
# sessions.json y dejan la VM inalcanzable—. Un relanzamiento diferido que no sabe qué pasó
# mientras tanto es una bomba de tiempo: cada script tiene que ser dueño de una sola cosa.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
LOGS="$AQUI/logs_campania_$(date +%Y%m%d)"; mkdir -p "$LOGS"

echo "== esperando a que termine el tramo huérfano de F (n4_s0)"
while pgrep -f "tramo_colab[.]sh F q_f_1" >/dev/null 2>&1; do sleep 30; done
echo "== tramo de F terminado · paso actual: $(
  /home/maxi/.venv-ligamento/bin/python -c "import pickle;print(pickle.load(open('$AQUI/ckpts/n4_s0.pkl','rb'))['paso'])" 2>/dev/null || echo '?')"

export CLOUDSDK_CONFIG="$HOME/.gcloud-cuentaF"
timeout 180 "$COLAB" --auth adc --config "$HOME/.colab-cuentaF.json" stop -s q_f_1 >/dev/null 2>&1 || true
unset CLOUDSDK_CONFIG
rm -f "$AQUI/claims/n4_s0"
echo "== VM de F apagada y n4_s0 liberada"

# Guarda: si por lo que sea ya hay un worker de F dando vueltas, no se agrega otro.
if pgrep -f "worker_cola[.]sh F " >/dev/null 2>&1; then
  echo "== ya hay un worker F vivo; no se relanza"
else
  nohup "$AQUI/worker_cola.sh" F 12000 4000 1000 >> "$LOGS/cola_F.log" 2>&1 &
  echo "== worker F relanzado (pid $!)"
fi
wait
