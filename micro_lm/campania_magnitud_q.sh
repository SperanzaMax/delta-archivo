#!/usr/bin/env bash
# CAMPANIA · la constante `q`, ¿es un problema de MAGNITUD?  ·  PREREG_MAGNITUD_Q.md (SHA 44c550e2)
#
#   Uso local:   ./campania_magnitud_q.sh <unidad> [pasos]
#   Uso Colab:   ver el reparto de abajo; cada cuenta corre 2 unidades
#
# Seis unidades: --rec-ce en {1.0 control, 0.50, 0.29} x origen en {b3_s3, b3_s6}.
# El 0.29 es 1/3,5, o sea el valor que iguala el ratio de gradientes MEDIDO en
# INFORME_RECOMPENSA_L_20260830.md. No es un barrido exploratorio.
#
# El resto queda fijo e igual a la campania de `L`, para que el unico factor que cambia sea el que se
# quiere medir: interfaz `token`, M=0.5, F=0.2, L=0, horizonte 12000.
#
# `--ckpt` guarda en CADA evaluacion y reanuda si el archivo ya existe, asi que una sesion de Colab
# que se muere a los ~60 min no pierde nada: se relanza el mismo comando y sigue.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASOS="${2:-12000}"
SAL="$AQUI/corridas_magnitud_q"
CK="$AQUI/ckpts_magnitud_q"
mkdir -p "$SAL" "$CK"

# unidad -> "rec_ce origen"
declare -A U=(
  [q10_s3]="1.0  b3_s3"      # control
  [q05_s3]="0.50 b3_s3"
  [q03_s3]="0.29 b3_s3"
  [q10_s6]="1.0  b3_s6"      # control
  [q05_s6]="0.50 b3_s6"
  [q03_s6]="0.29 b3_s6"
)

# Reparto sugerido para Colab, una cuenta por par. El control y su tratada NUNCA van juntas en la
# misma cuenta, para que ninguna diferencia entre cuentas caiga adentro del contraste.
#   A: q10_s3 q05_s6      C: q05_s3 q03_s6      D: q03_s3 q10_s6

correr() {
  local u="$1"; read -r ce origen <<< "${U[$u]}"
  local sem; sem="$(echo "$origen" | tr -dc '0-9')"
  echo "=== $u  ·  rec-ce=$ce  ·  origen=$origen  ·  semilla=$sem  ·  $PASOS pasos"

  # SIEMBRA. Continuar `b3_sX` con otra perdida es BIFURCAR, no continuar, y las guardas de
  # entrenar.py abortan con razon. `sembrar.py` lo hace explicito: conserva los pesos, borra el
  # estado de Adam y deja declarado `sembrado_de` adentro del checkpoint. Si ya existe no se re-siembra,
  # asi una sesion de Colab caida se relanza con el mismo comando y REANUDA en vez de empezar de cero.
  if [ ! -f "$CK/$u.pkl" ]; then
    python "$AQUI/sembrar.py" "$AQUI/ckpts/$origen.pkl" "$CK/$u.pkl" --horizonte 12000 \
      2>&1 | tee -a "$SAL/$u.log"
  else
    echo "  ($u ya tiene checkpoint, se REANUDA)" | tee -a "$SAL/$u.log"
  fi

  python "$AQUI/entrenar.py" \
      --abst token --rec-ce "$ce" --rec-l 0 --rec-m 0.5 --rec-f 0.2 --d 128 --capas 4 \
      --pasos "$PASOS" --horizonte 12000 --semilla "$sem" \
      --ckpt "$CK/$u.pkl" \
      --salida "$SAL/$u.json" \
      2>&1 | tee -a "$SAL/$u.log"
}

if [ $# -ge 1 ] && [ -n "${U[$1]:-}" ]; then
  correr "$1"
else
  echo "unidades disponibles: ${!U[@]}"
  echo "uso: $0 <unidad> [pasos]"
  exit 1
fi
