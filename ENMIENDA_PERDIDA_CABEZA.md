# ENMIENDA a `PREREG_PERDIDA_CABEZA.md` (SHA `0f57609d`)

**2026-08-29, antes de correr un solo paso.** El pre-registro se congeló a las 16:02 y **no se lanzó
nada**. Esta enmienda se escribe antes del primer tramo, y su motivo se descubrió **mientras se
preparaba el lanzamiento**, no después de ver resultados.

---

## 1. Qué se descubrió, y por qué obliga a enmendar

Al buscar las bases de siembra para las nueve semillas apareció que **sólo existen tres**
(`n3_s0`, `n3_s1`, `n3_s2`), y que el rotador siembra automáticamente cuando la base existe
(`SEMBRAR=1` por defecto, `tramo_abst.sh:88`). Medido hoy, esas bases entran a la fase de abstención
con la recuperación **ya aprendida**:

| base | pasos previos | RECUP al arrancar |
|---|---:|---:|
| n3_s0 | 12000 | 0,7734 |
| n3_s1 | 12000 | 0,7970 |
| n3_s2 | 12000 | 0,7725 |

**Las nueve unidades de control no son homogéneas.** `b3_s0/s1/s2` arrancaron desde ese punto y
`b3_s3`…`b3_s8` arrancaron de cero, y el desenlace se alinea con esa división casi perfectamente:

| grupo | punto de partida | resultado |
|---|---|---|
| s0, s1, s2 | RECUP ≈ 0,78 | **3 de 3 útiles** |
| s3 … s8 | de cero | 2 útiles, **4 mudas** |

El §3 del pre-registro dice **«flags idénticos a la campaña de control»** y el §4 construye P-0 y P-1
sobre «las cinco semillas que el control lleva a un estado útil» y «las cuatro que deja mudas».
**Ese reparto mezcla dos poblaciones distintas**, así que los criterios tal como están escritos
compararían la pérdida nueva contra un control que no es uno solo.

## 2. Lo que se cambia

**E-1 · No se siembra ninguna unidad.** Las dos condiciones corren con `SEMBRAR=0`, de cero, en
todas. Se declara en el comando y queda en el log del rotador.

**E-2 · El control pasa a ser `b3_s3`…`b3_s8`, las SEIS sin base.** Se descartan `b3_s0/s1/s2` como
comparación, porque partieron de otro lugar. El pareo queda exacto: **mismas seis semillas, misma
ausencia de base, y lo único que cambia es la pérdida de la cabeza.**

**E-3 · Las semillas de la campaña pasan de nueve a seis (3 a 8).** Correr 0, 1 y 2 sin sembrar daría
tres unidades de tratamiento **sin control pareado**, porque su control existente sí fue sembrado.
Costo: $6 \times 2 \times 3000 = 36000$ pasos, **menos** que los 54000 del diseño original.

**E-4 · P-0 se evalúa sobre dos unidades y eso lo debilita, dicho ahora.** El control deja en estado
útil sólo a `s4` y `s5` dentro del grupo sin base. El criterio de no-daño pasa a ser:

> **P-0 (enmendado).** Ninguna de `s4` y `s5` puede terminar en abstención total con la pérdida nueva.
> **Se permite cero de dos.**

Con $n=2$ esto detecta un daño grosero y nada más. **No se compensa aflojando otra cosa**, y si una
condición gana P-1 con P-0 al límite, el informe tiene que decir que el no-daño se apoyó en dos
unidades.

**E-5 · P-1 no cambia.** Sigue pidiendo **≥ 3 de 4** sobre `s3`, `s6`, `s7`, `s8`, que son
exactamente las cuatro mudas del grupo sin base. El número no se toca, y era el mismo antes de saber
esto.

**Lo demás del pre-registro queda igual**, incluidos P-2, P-3, P-4, P-5, la tabla de desenlaces del
§5 y el criterio de abandono del §6.

## 3. Lo que esta enmienda NO hace

- **No cambia ningún umbral hacia abajo.** P-1 sigue en 3 de 4 y P-3 sigue en AUC 0,60. Lo único que
  se movió fue **de qué población se cuenta**, y se movió hacia el control más estricto disponible,
  no hacia el más cómodo.
- **No rescata el pre-registro original.** El defecto era del control heredado, no del diseño de las
  pérdidas, y la compuerta `chequeo_perdida_cabeza.py` ya había abierto con las tres condiciones
  verificadas antes de que esto apareciera.
- **No corrige el hallazgo del 29.** Eso se hace donde corresponde, en
  `INFORME_ATRACTOR_MUDO_FASE1_20260829.md` y en el manuscrito, y ya está hecho.

## 4. Lección de método, para que no se repita

> **La config guardada no registra si hubo siembra.** El chequeo del 28 comparó configs y concluyó
> que las unidades sólo diferían en la semilla; era cierto de la config y falso del experimento.
> De acá en adelante, **toda verificación de homogeneidad tiene que mirar el punto de partida y no
> sólo los flags**, y el arnés debería anotar el origen del checkpoint inicial en la propia config.
