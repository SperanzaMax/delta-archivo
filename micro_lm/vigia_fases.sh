#!/usr/bin/env bash
# Avisa por Telegram cada fase de la frontera que CIERRA, ya con su veredicto de compuerta.
#
# Va aparte de los avisos de `rotar_frontera.sh` a proposito: aquellos dicen «tramo cerrado en el
# paso N», que es estado de la infra. Este dice si la unidad PASO o FALLO, que es el resultado.
#
# Corre por systemd --user y no dentro de la sesion de trabajo: el 2026-08-18 la sesion murio por
# systemd-oomd y el 19 el escritorio se colgo 69 minutos (ver la memoria de termica). El aviso tiene
# que sobrevivir a las dos cosas, asi que no puede depender de que haya alguien mirando.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SALIDA="$AQUI/corridas_$(date +%Y%m%d)"
ESTADO="$AQUI/.avisadas_fases"
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo
touch "$ESTADO"

mandar() {
  curl -s -m 20 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null 2>&1
}

ya_avisada() { grep -qx "$1" "$ESTADO"; }
marcar()     { echo "$1" >> "$ESTADO"; }

# --- las 12 fases: se avisa la que llego a su paso final -----------------------------------------
# El paso final es distinto en cada una (el corte cayo donde `vigente` cruzo el valor, no en un paso
# redondo), asi que sale de fases.tsv y no de una constante.
[ -f "$AQUI/fases.tsv" ] && while IFS=$'\t' read -r uni base pbase ptot abst vcorte; do
  case "$uni" in \#*|"") continue;; esac
  ya_avisada "$uni" && continue
  JS="$SALIDA/${uni}.json"
  [ -f "$JS" ] || continue
  LINEA="$(python3 - "$JS" "$ptot" <<'PY'
import json, sys, math
d = json.load(open(sys.argv[1]))
ult = [h for h in d["historia"] if h["paso"] == int(sys.argv[2])]
if not ult:
    sys.exit(1)
h = ult[-1]
n, fa, v = h.get("nose"), h.get("falsa_abst"), h.get("vigente")
if n is None or (isinstance(n, float) and math.isnan(n)):
    sys.exit(1)
ok = "PASA" if (n >= 0.50 and fa <= 0.10) else "falla"
print(f"{v:.4f}|{n:.4f}|{fa:.4f}|{ok}")
PY
)" || continue
  IFS='|' read -r V N FA OK <<< "$LINEA"
  mandar "micro-LM · frontera · ${uni} CERRÓ (${abst}, margen de entrada ${vcorte})
   vigente ${V} · nose ${N} · falsa_abst ${FA}
   compuerta: ${OK}"
  marcar "$uni"
done < "$AQUI/fases.tsv"

# --- f2_s1: la semilla trabada. Interesa el CRUCE de 0,85, que es lo que habilita sus 6 fases ----
if ! ya_avisada "f2_s1_cruce" && [ -f "$SALIDA/f2_s1.json" ]; then
  CRUCE="$(python3 - "$SALIDA/f2_s1.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for h in d["historia"]:
    if h["vigente"] >= 0.85:
        print(f'{h["paso"]}|{h["vigente"]:.4f}')
        break
PY
)"
  if [ -n "$CRUCE" ]; then
    IFS='|' read -r P V <<< "$CRUCE"
    mandar "micro-LM · frontera · f2_s1 POR FIN CRUZÓ 0,85 en el paso ${P} (vigente ${V}).
Se estancaba en 0,7777 a 6000 pasos. Ya se pueden correr sus 6 fases y la campaña queda con las 3 semillas que pedía el diseño."
    marcar "f2_s1_cruce"
  fi
fi

# --- f2_s1 agotado sin cruzar: tambien es una respuesta, y hay que decirla ------------------------
if ! ya_avisada "f2_s1_fin" && [ -f "$SALIDA/f2_s1.json" ]; then
  if grep -q '"paso": 14000' "$SALIDA/f2_s1.json" 2>/dev/null && ! ya_avisada "f2_s1_cruce"; then
    mandar "micro-LM · frontera · f2_s1 llegó a 14000 pasos SIN cruzar 0,85.
Se queda en la meseta de ~0,77. La campaña de la frontera cierra con 2 semillas, no 3, y eso va declarado en las desviaciones."
    marcar "f2_s1_fin"
  fi
fi
