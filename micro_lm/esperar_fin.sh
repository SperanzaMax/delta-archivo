#!/usr/bin/env bash
# Espera a que TERMINE todo lo que quedó corriendo la noche del 17-ago y avisa por Telegram.
# Sale con código 0 cuando no queda nada pendiente, o 2 si se agota el plazo.
#
#   Uso:  esperar_fin.sh [minutos_max]
#
# Existe porque el cierre de la jornada depende de que las últimas corridas lleguen a su paso final,
# y quedarse mirando un log a mano no escala. Sólo lee archivos locales: nunca toca el CLI de Colab,
# que es de los rotadores.
set -uo pipefail

MAX_MIN="${1:-150}"
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo

# unidad:paso_objetivo
PENDIENTES=(x3_s0:14000 x2_s0:14000 x1_s1:13000)

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

paso_de() {
  grep -o '"paso": [0-9]*' "$SALIDA/$1.json" 2>/dev/null | tail -1 | grep -o '[0-9]*'
}

listo() {
  local u="${1%%:*}" meta="${1##*:}" p
  p="$(paso_de "$u")"
  [ -n "$p" ] && [ "$p" -ge "$meta" ]
}

echo "[$(date +%H:%M)] esperando: ${PENDIENTES[*]} (máx ${MAX_MIN} min)"
FIN=$(( $(date +%s) + MAX_MIN * 60 ))
declare -A avisadas=()

while [ "$(date +%s)" -lt "$FIN" ]; do
  faltan=()
  for u in "${PENDIENTES[@]}"; do
    nom="${u%%:*}"
    if listo "$u"; then
      # aviso individual, una sola vez por unidad
      if [ -z "${avisadas[$nom]:-}" ]; then
        avisadas[$nom]=1
        det="$(python3 - "$SALIDA/$nom.json" <<'PY' 2>/dev/null
import json,sys
d=json.loads(open(sys.argv[1]).read().replace('NaN','null'))
h=d['historia'][-1]
n=h.get('nose'); f=h.get('falsa_abst')
if n is None:
    print(f"paso {h['paso']} · vigente {h['vigente']:.4f}")
else:
    ok = 'PASA la compuerta' if (n>=0.50 and f<=0.10) else 'no pasa'
    print(f"paso {h['paso']} · vigente {h['vigente']:.4f} · nose {n:.4f} · falsa_abst {f:.4f} → {ok}")
PY
)"
        echo "[$(date +%H:%M)] terminó $nom: $det"
        mandar "✔️ micro-LM · $nom terminó.
$det"
      fi
    else
      faltan+=("$nom($(paso_de "$nom" || echo '-'))")
    fi
  done

  if [ "${#faltan[@]}" -eq 0 ]; then
    echo "[$(date +%H:%M)] TODO TERMINADO"
    mandar "🏁 micro-LM · terminó TODO lo que quedaba corriendo. Listo para el cierre de la jornada."
    exit 0
  fi
  sleep 120
done

echo "[$(date +%H:%M)] plazo agotado; seguían: ${faltan[*]}"
mandar "⏰ micro-LM · se cumplieron ${MAX_MIN} min de espera y todavía seguían: ${faltan[*]}"
exit 2
