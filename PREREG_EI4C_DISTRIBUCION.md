# PREREG · E-I4c — FORZAR LA DERIVA POR CAMBIO DE DISTRIBUCIÓN

Congelado el 2026-08-21 antes de escribir el script. SHA en `SHA_EI4C.txt`.

Antecedentes directos: `INFORME_EI4_20260813.md` y `INFORME_EI4B_20260813.md`. Los dos midieron el
envejecimiento del archivo por **antigüedad** (el archivo se escribe con pesos viejos y se lee con los
de hoy) y los dos dejaron la pregunta central sin responder, por la misma razón.

## §1 · Por qué hace falta, y qué queda exactamente sin contestar

R5.1 midió afuera, con encoder congelado e índice no paramétrico, que la memoria persistente funciona
mientras `cos(marco de hoy, marco de escritura) ≳ 0,70`, degrada entre 0,70 y 0,40 y muere debajo. La
pregunta que abre el brazo interno es si **un índice co-entrenado tolera lo que mata al no
paramétrico** — y ésa es la pregunta que sigue abierta:

- **E-I4** (edades 0-400) no encontró degradación **porque el coseno no bajó de 0,9374**: midió la
  zona donde la propia teoría del proyecto predice que no pasa nada. Negativo con estímulo débil.
- **E-I4b** (edades 0-8000, 12000 pasos) llegó a cos 0,7804 y encontró pendiente que se empina, sin
  acantilado en 0,70. Pero **P-2 quedó NO EVALUABLE: el coseno nunca cruzó 0,70.**

El propio informe de E-I4b dejó escrita la salida, y es la que se ejecuta acá: *«hacen falta edades de
16000-32000 pasos (el marco se mueve cada vez más despacio a medida que el modelo converge) o forzar
la deriva con cambio de distribución, que es lo que hizo R6 afuera. La segunda vía es más barata y
probablemente más realista: un modelo desplegado no envejece por pasos de gradiente, envejece porque
lo siguen afinando en datos nuevos.»*

## §2 · El cambio de distribución, y por qué esta forma

**Partición del vocabulario de claves.** `V_E001` tiene NK = 128 claves. La fase A entrena sorteando
claves de `[0, 64)`; la fase B sigue entrenando el MISMO modelo sorteando de `[64, 128)`. El archivo
se escribe con los pesos del final de la fase A y se lee con los pesos de la fase B.

Tres razones para elegir ésta y no otra:

1. **No cambia ninguna forma de tensor.** La alternativa obvia —cambiar la carga `L`, que es lo que
   hizo R6 afuera— mueve `N_ARCH = L + R` y con eso el tamaño del archivo y el uso de `ord`. Ahí un
   resultado negativo sería ininterpretable: no se sabría si degradó la deriva o el cambio de forma.
2. **Es el análogo interno de R6**: distribución nueva sobre un modelo ya entrenado, que es donde R6
   encontró que el preentrenado aguanta (cos 0,882 contra 0,207 desde cero).
3. **Es realista en el sentido que importa para la línea**: el archivo guarda hechos viejos mientras
   el modelo sigue aprendiendo sobre cosas nuevas. Es literalmente el caso de uso.

## §3 · Instrumento

Se reusa `interno/ei4_envejecimiento.py` (escritura con pesos viejos, lectura con los de hoy, coseno
sobre las claves archivadas) y `interno/ei3_orden.py` (modo `sello`, que es la configuración que
funciona). Lo único nuevo es el generador por fases.

- Fase A: 6000 pasos con claves de `[0, 64)`. La tarea converge cerca de 3000 (E-I4b), así que llega
  entrenada.
- Fase B: se miden **edades** 0 · 500 · 2000 · 6000 pasos dentro de la fase B.
- 3 semillas, reportadas **por semilla** y no sólo en media.

## §4 · Predicciones

- **P-1 (BLOQUEANTE, control del instrumento).** El coseno cruza **0,70** en la edad máxima. Si no lo
  cruza, este experimento **tampoco** tiene poder de resolución y se declara así, en vez de leerse
  como robustez. Es la misma cláusula que E-I4b puso y que salvó a E-I4 de una lectura equivocada.
  Que la cláusula ya haya disparado una vez es la razón de mantenerla.
- **P-2 (LA PREGUNTA).** Con cos < 0,70, la accuracy en revisadas cae **≥ 0,10** respecto de la edad 0.
  - Si cae → el índice co-entrenado se comporta como el no paramétrico y el umbral de R5.1 es una
    propiedad del problema, no del método.
  - **Si NO cae → resultado positivo fuerte y nuevo**: el índice co-entrenado tolera lo que mata al no
    paramétrico. Ésa es la primera cosa buena que diría el brazo interno sobre persistencia, y por eso
    la predicción se declara en la dirección que me haría equivocar.
- **P-3 (tasa de deriva).** El cambio de distribución mueve el marco **más rápido por paso** que la
  antigüedad pura de E-I4b: cos a 2000 pasos de fase B < 0,9067 (el valor de E-I4b a edad 2000). Es lo
  que justifica haber elegido esta vía por sobre entrenar 32000 pasos.

## §5 · Regla de cierre

- Si **P-1 falla**, se cierra la vía del envejecimiento **entera** —las dos formas de producir deriva
  ya habrían fallado en producirla— y se reporta como límite del harness, no del mecanismo.
- Si P-1 pasa y P-2 se resuelve en cualquiera de las dos direcciones, la pregunta de E-I4 queda
  **contestada** y el envejecimiento sale de la lista de pendientes.

## §6 · Lo que no puede decir

- Un solo tipo de cambio de distribución. Que el marco se mueva al cambiar las claves no dice qué pasa
  al cambiar la estructura de la tarea.
- La accuracy de la fase B se mide sobre hechos de la distribución A, que es lo correcto para la
  pregunta —el archivo es viejo—, pero significa que el modelo está siendo evaluado fuera de lo que
  entrena en la fase B. **El control que lo separa es la edad 0**: mismo desajuste de distribución,
  sin deriva acumulada.
