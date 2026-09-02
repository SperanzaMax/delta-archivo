#!/usr/bin/env bash
# Estado de TODO lo que corre, en un solo bloque de texto listo para Telegram.
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=/home/maxi/.venv-ligamento/bin/python
paso(){ $PY -c "import pickle;print(pickle.load(open('$AQUI/ckpts/$1.pkl','rb')).get('paso',0))" 2>/dev/null || echo 0; }
barra(){ local n=$1 t=26000; local l=$((n*12/t)); printf '%s%s %5d/%d' "$(printf '█%.0s' $(seq 1 $((l>0?l:1))))" "$(printf '·%.0s' $(seq 1 $((12-l>0?12-l:0))))" "$n" "$t"; }

echo "MICRO-LM"
echo "  kernel 7 (¿más ventana ensucia?)"
for s in 0 1 2; do printf '    k73_s%s  %s\n' "$s" "$(barra "$(paso k73_s$s)")"; done
echo "  el CRUCE (dos formas, la relación entra a veces)"
for s in 1 2 3; do printf '    cf3_s%s  %s\n' "$s" "$(barra "$(paso cf3_s$s)")"; done
echo "  el CONTROL ciego (dos formas, la relación NUNCA entra)"
for s in 0 1 2; do printf '    cl3_s%s  %s\n' "$s" "$(barra "$(paso cl3_s$s)")"; done

echo
echo "MODELO REAL · mamba-370m en T4"
for f in "$AQUI"/../modelo_real/real_*_run.log; do
  [ -f "$f" ] || continue
  u="$(basename "$f" _run.log)"
  ult="$(grep -oE "eval [0-9]+: .*|BASELINE paso 0 · .*" "$f" 2>/dev/null | tail -1)"
  est="$(grep -c "VIVO= False" "$f" 2>/dev/null)"
  printf '  %-14s %s\n' "$u" "${ult:-arrancando}"
done

echo
vivos=$(ps aux | grep -c "[r]otar_abst3")
echo "rotadores vivos $vivos · $(date +%H:%M)"
