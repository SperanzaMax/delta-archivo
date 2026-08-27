#!/usr/bin/env bash
# Panel del estado de TODA la jornada: campaña base (familia "n") y abstención (familia "x").
#
#   Uso:  panel.sh            -> imprime el panel
#         panel.sh telegram   -> lo manda al chat de Maxi
#         panel.sh bucle 30   -> lo manda cada 30 minutos
#
# Reemplaza a `reporte30.sh`, que tenia la familia "n" escrita a mano y por eso no veia ninguna de
# las corridas de abstencion. Como aquel, SOLO LEE ARCHIVOS LOCALES: mientras un rotador pollea una
# cuenta, otro proceso `colab` sobre esa misma cuenta se pisa el sessions.json y deja la VM
# inalcanzable. El estado ya viaja a la PC por streaming, asi que mirar el disco alcanza.
#
# La compuerta de abstencion es `nose >= 0,50` Y `falsa_abst <= 0,10`: las dos juntas, porque el que
# se abstiene de TODO saca nose = 1 y hay que poder rechazarlo (test_metricas_nose.py lo verifica).
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"   # TOKEN y CHAT salen de fuera del repo
ATAJO=0.5906   # acierto de "no abstenerse nunca" con p_nose = 0,4

panel() {
python3 - "$AQUI" <<'PY'
import json, os, sys, glob
aqui = sys.argv[1]
def leer(uni):
    mejor = None
    for f in sorted(glob.glob(os.path.join(aqui, 'corridas_*', uni + '.json'))):
        try:
            d = json.loads(open(f).read().replace('NaN', 'null'))
        except Exception:
            continue
        h = [x for x in d.get('historia', []) if x.get('paso') is not None]
        if not h:
            continue
        if mejor is None or h[-1]['paso'] >= mejor[0]['paso']:
            mejor = (h[-1], d.get('config', {}), d.get('hw', '?'))
    return mejor

print('CAMPAÑA BASE (12000 pasos)')
hechas, faltan = [], []
for n in (1, 2, 3, 4):
    for s in (0, 1, 2):
        r = leer(f'n{n}_s{s}')
        if r is None:
            faltan.append(f'n{n}_s{s} (sin datos)'); continue
        u, _, _ = r
        (hechas if u['paso'] >= 12000 else faltan).append(
            f"n{n}_s{s} {u['paso']}" + ('' if u['paso'] >= 12000 else f" ({u['vigente']:.4f})"))
print(f'  ✅ {len(hechas)}/12 cerradas')
if faltan:
    print('  ⏳ ' + ' · '.join(faltan))

print()
print('ABSTENCIÓN · compuerta = nose ≥ 0,50 Y falsa_abst ≤ 0,10')
print(f"  {'unidad':<8}{'margen':>8}{'paso':>7}{'vigente':>9}{'nose':>8}{'f_abst':>8}  estado")
filas = []
for n in (1, 2, 3, 4):
    for s in (0, 1, 2):
        r = leer(f'x{n}_s{s}')
        if r is None:
            continue
        u, cfg, hw = r
        base = leer(f'n{n}_s{s}')
        margen = (base[0]['vigente'] - 0.5906) if base and base[0]['paso'] >= 12000 else None
        if u.get('nose') is None:
            estado = 'aún sin NOSE'
            filas.append((margen, f"  x{n}_s{s:<6}{(f'{margen:+.4f}' if margen is not None else '   ?  '):>8}"
                                  f"{u['paso']:>7}{u['vigente']:>9.4f}{'-':>8}{'-':>8}  {estado}"))
            continue
        pasa = u['nose'] >= 0.50 and u['falsa_abst'] <= 0.10
        estado = 'PASA' if pasa else ('falla: se calla de más' if u['falsa_abst'] > 0.10 else 'falla: no se abstiene')
        if u['vigente'] < 0.5906:
            estado += ' · POR DEBAJO DEL ATAJO'
        filas.append((margen, f"  x{n}_s{s:<6}{(f'{margen:+.4f}' if margen is not None else '   ?  '):>8}"
                              f"{u['paso']:>7}{u['vigente']:>9.4f}{u['nose']:>8.4f}{u['falsa_abst']:>8.4f}  {estado}"))
for _, f in sorted(filas, key=lambda t: -(t[0] if t[0] is not None else -9)):
    print(f)
PY
}

CUERPO="$(panel)"
T="$(sensors 2>/dev/null | sed 's/(.*//' | grep -oE '\+[0-9]+\.[0-9]+°C' | tr -d '+°C' | sort -rn | head -1 | cut -d. -f1)"
ROT="$(pgrep -c -f 'rotar[_]tramos' 2>/dev/null || echo 0)"
PIE="$(printf 'rotadores vivos: %s · CPU %s °C · %s' "$ROT" "${T:-?}" "$(date +%H:%M)")"

case "${1:-ver}" in
  telegram)
    curl -s -m 25 -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" \
      --data-urlencode "text=📊 micro-LM
$CUERPO

$PIE" >/dev/null ;;
  bucle)
    MIN="${2:-30}"
    while true; do "$0" telegram; sleep $(( MIN * 60 )); done ;;
  *)
    echo "$CUERPO"; echo; echo "$PIE" ;;
esac
