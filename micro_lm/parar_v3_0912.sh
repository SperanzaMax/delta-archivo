#!/usr/bin/env bash
# 12-sep 11:05 · parar la v3 de encadenados (mc3/md3: atajo del orden, E-4) SIN tocar tw3_s0 (cuenta E)
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
for u in mc3_s1 mc3_s2 md3_s1 md3_s2; do
  for p in $(pgrep -f "rotador_${u}.log"); do kill "$p" 2>/dev/null; done
done
sleep 1
# tramos vivos de esas unidades (el rotador los lanza como hijos; pueden quedar huerfanos)
for p in $(pgrep -f "tramo_abst.sh"); do
  if tr '\0' ' ' < /proc/$p/environ 2>/dev/null | grep -qE "PREFIJO=m[cd]"; then pkill -TERM -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; echo "tramo $p parado"; fi
done
for C in C D F G I J L N; do
  CL=( env CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$C" "$COLAB" --auth adc --config "$HOME/.colab-cuenta$C.json" )
  S="$(timeout 60 "${CL[@]}" sessions 2>&1 | grep -aoE "^\[tr2_[a-z]_[0-9]+\]" | tr -d '[]' | head -1)"
  [ -z "$S" ] && continue
  U="$(grep -al "sesion $S" "$AQUI"/corridas_20260912/rotador_m*.log 2>/dev/null | tail -1 | sed -E 's/.*rotador_(.*)\.log/\1/')"
  [ -z "$U" ] && { echo "-- $C / $S no es de mc/md, se deja"; continue; }
  timeout -k 30 120 "${CL[@]}" stop -s "$S" >/dev/null 2>&1 && echo "-- $C / $S ($U) parada"
  rm -f "$HOME/.colab-pool/en_uso_$C"
done
pgrep -af "rotar_abst3" | grep -v pgrep | cut -c1-100
