#!/usr/bin/env bash
# CAMPANIA · sello RELATIVO + bit de pertenencia  ·  PREREG_SELLO_RELATIVO.md (SHA ae66e464)
#
#   Uso local:   ./campania_sello_relativo.sh <unidad> [pasos]
#
# SEIS unidades, no nueve: el control `abs` YA esta corrido (lg3_s0/s1/s2, campania del archivo
# largo del 5-sep), asi que la GPU se gasta solo en lo que falta.
#
#   rp3_sN  --sello rel --pert   la principal: las dos piezas juntas
#   rr3_sN  --sello rel          aisla cuanto aporta el sello relativo SOLO
#
# De donde sale: el 6-sep quedo medido, por conducta (`aabb4b20`) y por pesos (`bedac0b5`), que el
# sello absoluto hace que el modelo aprenda una BANDERA — una marca de «ajeno» copiada en las 24
# filas del bloque bajo— en vez de un reloj. Correr el episodio dos lugares, con el orden relativo
# intacto, tira el acierto de 1,00 a 0,02-0,08.
#
# Mismo presupuesto y misma siembra que la campania del archivo largo: 2000 pasos con horizonte
# 6000, desde kq3_sN, con --ses-extra 26 (300 casilleros, ~161 entradas).
#
# MICRO-LOTES OBLIGATORIOS: con 30 sesiones y batch 64 la T4 pide 63,73 GiB y muere (medido el
# 5-sep). `--micro-batch 8` promedia gradientes y deja el batch EFECTIVO igual (verificado en
# 1,49e-08 de diferencia relativa).
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASOS="${2:-2000}"
HOR="${HOR:-6000}"
SAL="$AQUI/corridas_sello_rel"
CK="$AQUI/ckpts_sello_rel"
mkdir -p "$SAL" "$CK"

# unidad -> "flags_de_sello origen"
declare -A U=(
  [rp3_s0]="--sello rel --pert|kq3_s0"
  [rp3_s1]="--sello rel --pert|kq3_s1"
  [rp3_s2]="--sello rel --pert|kq3_s2"
  [rr3_s0]="--sello rel|kq3_s0"
  [rr3_s1]="--sello rel|kq3_s1"
  [rr3_s2]="--sello rel|kq3_s2"
)

# Reparto sugerido, una cuenta por par. Las dos condiciones de una misma semilla NUNCA van juntas en
# la misma cuenta, para que ninguna diferencia entre cuentas caiga adentro del contraste.
#   A: rp3_s0 rr3_s1      C: rp3_s1 rr3_s2      D: rp3_s2 rr3_s0

correr() {
  local u="$1"
  local flags="${U[$u]%%|*}" origen="${U[$u]##*|}"
  local sem; sem="$(echo "$origen" | tr -dc '0-9' | tail -c 1)"
  echo "=== $u  ·  $flags  ·  origen=$origen  ·  semilla=$sem  ·  $PASOS pasos (horizonte $HOR)"

  # SIEMBRA. Cambiar el modo del sello ES bifurcar y la guarda de `entrenar.py` aborta con razon.
  # `sembrar.py` lo hace explicito: conserva los pesos, borra el estado de Adam, saca `sello`/`pert`
  # /`ses_extra` de la config y —desde el 6-sep— AGREGA `pert` en cero si el checkpoint es anterior.
  # Sin eso la campania correria con `--pert` puesto y el bit no entraria nunca, en silencio.
  if [ ! -f "$CK/$u.pkl" ]; then
    python "$AQUI/sembrar.py" "$AQUI/ckpts/$origen.pkl" "$CK/$u.pkl" --horizonte "$HOR" \
      2>&1 | tee -a "$SAL/$u.log"
  else
    echo "  ($u ya tiene checkpoint, se REANUDA)" | tee -a "$SAL/$u.log"
  fi

  # shellcheck disable=SC2086
  python "$AQUI/entrenar.py" \
      $flags \
      --ses-extra 26 --kernel-q 5 --donde lat2 --nivel 3 --d 128 --capas 4 \
      --pasos "$PASOS" --horizonte "$HOR" --semilla "$sem" \
      --p-nose 0.2 --micro-batch 8 --batch-eval 8 \
      --ckpt "$CK/$u.pkl" \
      --salida "$SAL/$u.json" \
      2>&1 | tee -a "$SAL/$u.log"

  # Si el entrenamiento no llego, NO se mide: un JSON de metricas sobre un checkpoint que no
  # entreno es exactamente la clase de numero que despues se lee como resultado. Lo aprendio el
  # smoke del 6-sep, que midio alegremente sobre la siembra despues de un ABORTA.
  if ! grep -q "listo:" "$SAL/$u.log"; then
    echo "!! $u NO termino de entrenar: no se mide. Ver $SAL/$u.log"
    return 1
  fi

  # R-1 y R-3 se miden sobre el checkpoint terminado, en la misma corrida que lo produjo.
  python "$AQUI/reloj_o_bandera.py" "$CK/$u.pkl" --lotes 4 --batch 16 \
      --salida "$SAL/${u}_reloj.json" 2>&1 | tee -a "$SAL/$u.log"
  python "$AQUI/geometria_ord.py" "$CK/$u.pkl" \
      --salida "$SAL/${u}_geom.json" 2>&1 | tee -a "$SAL/$u.log"
}

if [ $# -ge 1 ] && [ -n "${U[$1]:-}" ]; then
  correr "$1"
else
  echo "unidades disponibles: ${!U[@]}"
  echo "uso: $0 <unidad> [pasos]"
  exit 1
fi
