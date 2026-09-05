#!/usr/bin/env bash
# CAMPANIA · ¿aprende a buscar en un archivo largo?  ·  PREREG_ARCHIVO_LARGO.md (SHA c769a4ef)
#
#   Uso local:   ./campania_archivo_largo.sh <unidad> [pasos]
#   Uso Colab:   ver el reparto de abajo
#
# Seis unidades: --ses-extra en {0 control, 26 tratada} x origen en {kq3_s0, kq3_s1, kq3_s2}.
# Los kq3 son las tres unidades de kernel 5 a 26.000 pasos que YA resuelven la tarea con archivo
# corto (0,988-0,993), asi que la pregunta es si un modelo que la sabe aprende a hacerla con un
# archivo grande — no si puede aprenderla.
#
# --ses-extra 26 lleva el archivo de 40 a 300 casilleros (~161 entradas escritas), que es el regimen
# donde la exactitud medida el 5-sep da 0,3008. Costo medido: 6,96x por paso contra el control, o sea
# ~1,53 s/paso en T4 y ~2,5 h por unidad tratada; el control son ~0,4 h.
#
# El control NO es decorativo: separa «entrenar con archivo largo» de «entrenar 6000 pasos mas».
# Lleva la misma siembra y el mismo presupuesto.
#
# `--ckpt` guarda en CADA evaluacion y reanuda si el archivo ya existe: una sesion de Colab que se
# muere no pierde nada, se relanza el mismo comando y sigue.
set -uo pipefail

AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PASOS="${2:-6000}"
SAL="$AQUI/corridas_archivo_largo"
CK="$AQUI/ckpts_archivo_largo"
mkdir -p "$SAL" "$CK"

# unidad -> "ses_extra origen"
declare -A U=(
  [al0_s0]="0  kq3_s0"       # control
  [al26_s0]="26 kq3_s0"
  [al0_s1]="0  kq3_s1"       # control
  [al26_s1]="26 kq3_s1"
  [al0_s2]="0  kq3_s2"       # control
  [al26_s2]="26 kq3_s2"
)

# Reparto sugerido, una cuenta por par. El control y su tratada NUNCA van juntas en la misma cuenta,
# para que ninguna diferencia entre cuentas caiga adentro del contraste.
#   A: al26_s0 al0_s1      C: al26_s1 al0_s2      D: al26_s2 al0_s0

correr() {
  local u="$1"; read -r extra origen <<< "${U[$u]}"
  local sem; sem="$(echo "$origen" | tr -dc '0-9' | tail -c 1)"
  echo "=== $u  ·  ses-extra=$extra  ·  origen=$origen  ·  semilla=$sem  ·  $PASOS pasos"

  # SIEMBRA. Cambiar el tamanio del archivo es BIFURCAR: es otra tarea y la guarda de entrenar.py
  # aborta con razon. `sembrar.py` lo hace explicito —conserva los pesos, borra el estado de Adam,
  # saca `ses_extra` de la config y deja `sembrado_de` adentro del checkpoint—. Si ya existe no se
  # re-siembra, asi una sesion caida se relanza con el mismo comando y REANUDA.
  if [ ! -f "$CK/$u.pkl" ]; then
    python "$AQUI/sembrar.py" "$AQUI/ckpts/$origen.pkl" "$CK/$u.pkl" --horizonte "$PASOS" \
      2>&1 | tee -a "$SAL/$u.log"
  else
    echo "  ($u ya tiene checkpoint, se REANUDA)" | tee -a "$SAL/$u.log"
  fi

  python "$AQUI/entrenar.py" \
      --ses-extra "$extra" --kernel-q 5 --donde lat2 --nivel 3 --d 128 --capas 4 \
      --pasos "$PASOS" --horizonte "$PASOS" --semilla "$sem" \
      --p-nose 0.2 \
      --ckpt "$CK/$u.pkl" \
      --salida "$SAL/$u.json" \
      2>&1 | tee -a "$SAL/$u.log"

  # L-4 se mide sobre el checkpoint terminado, con su control de barajado. Va aca y no despues para
  # que quede en la misma corrida que lo produjo.
  python "$AQUI/masa_turnos.py" "$CK/$u.pkl" --ses-extra 26 --barajar --lotes 4 \
      --salida "$SAL/${u}_masa.json" 2>&1 | tee -a "$SAL/$u.log"
}

if [ $# -ge 1 ] && [ -n "${U[$1]:-}" ]; then
  correr "$1"
else
  echo "unidades disponibles: ${!U[@]}"
  echo "uso: $0 <unidad> [pasos]"
  exit 1
fi
