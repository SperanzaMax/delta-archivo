#!/usr/bin/env bash
# Vigia de la campania de `PREREG_PERDIDA_CABEZA.md`. Le manda a Maxi por Telegram, cada media hora,
# como viene cada unidad y —lo que de verdad importa— si SALIO del silencio.
#
#   setsid ./vigia_perdida.sh > vigia_perdida.log 2>&1 &
#
# El criterio P-1 se lee de un solo numero por unidad, `abstencion`. El control (b3_s3/s6/s7/s8) da
# 1,0000 en todos sus hitos hasta 26000, asi que cualquier valor por debajo de 1,0000 en estas
# unidades ya es la senial que la campania fue a buscar. Se marca con ★ para que se vea de un vistazo
# en el celular.
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/maxi/.venv-ligamento/bin/python}"
. "$AQUI/tg_token.sh"
CADA="${CADA_AVISO:-1800}"

mandar() {
  curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" \
    -d chat_id="$CHAT" --data-urlencode "text=$1" >/dev/null
}

foto() {
  "$PY" - <<'PY'
import json, glob, os, datetime
sal = f"micro-LM · perdida de cabeza · {datetime.datetime.now():%H:%M}\n"
sal += "control b3 = 4 de 4 MUDAS · balance y ranking = hablan pero INVENTAN\n"
sal += "(exactitud 0,236-0,402 contra un piso de 0,4065)\n\n"
for pre, nom in (("tk", "token (PRINCIPAL)"), ("hd", "cabeza")):
    sal += f"--- {nom} ---\n"
    hay = False
    for s in (3, 6, 7, 8, 4, 5):
        f = f"corridas_{datetime.date.today():%Y%m%d}/{pre}3_s{s}.json"
        if not os.path.exists(f):
            continue
        hay = True
        try:
            h = json.load(open(f))["historia"]
        except Exception:
            continue
        if not h:
            continue
        u = h[-1]
        # ¿alguna vez emitio algo?
        salio = [r["paso"] for r in h if r.get("abstencion", 1.0) < 1.0]
        marca = "★ HABLA" if salio else "  muda "
        rol = "P-1" if s in (3, 6, 7, 8) else "P-0"
        sal += (f"{marca} {pre}3_s{s} [{rol}] paso {u['paso']:5d} · abst {u.get('abstencion',0):.4f}"
                f" · vig {u.get('vigente',0):.4f}")
        if salio:
            sal += f" · primer habla en {salio[0]}"
        sal += "\n"
    if not hay:
        sal += "  (todavia sin datos)\n"
    sal += "\n"
sal += "W-1 pide >= 4 de 6 de token saliendo del silencio.\n"
sal += "OJO: el riesgo de esta campania es lo CONTRARIO, que hable de mas."
print(sal)
PY
}

cd "$AQUI"
while true; do
  # se corta solo cuando ya no queda ningun rotador de esta campania
  vivos=$(ps -eo args --no-headers | grep -c "[r]otar_abst3.sh 3:" || true)
  msg="$(foto)"
  if [ "$vivos" = "0" ]; then
    mandar "$msg

== TODOS LOS ROTADORES TERMINARON. Campaña cerrada, falta el veredicto. =="
    break
  fi
  mandar "$msg
(rotadores vivos $vivos)"
  sleep "$CADA"
done
