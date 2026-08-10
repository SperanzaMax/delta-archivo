# Enmienda E2 — Indexación: qué dirección y qué contenido lleva cada entrada

**Congelada el 2026-08-10, antes de generar los embeddings definitivos y antes de correr ninguna de
las cuatro condiciones.** Ninguna condición fue analizada todavía (ver `INFORME_FINAL_20260809.md`):
el experimento se detuvo dos veces por compuerta, así que no hay dato observado que pueda haber
influido en lo que sigue.

Enmienda a `PREREG_HECHOS.md` (SHA en `PREREG_HASH.txt`). Antecedentes: desviación D1
(`DESVIACIONES_HECHOS.md`) y enmienda E1 (`ENMIENDA_E1_HECHOS.md`, cambio de encoder).

---

## 1. Por qué hace falta

§3 del pre-registro define las condiciones así:

| condición | dirección de la revisión | guarda historia |
|---|---|---|
| `sin` | — (control, sin archivo) | no |
| `sobrescritura` | `emb(v2)`, reemplaza la entrada de v1 | no |
| `duplicados` | `emb(v2)`, entrada nueva independiente | sí, sin estructura |
| `gemacion` | `emb(v1) + ε·t̂`, entrada nueva anclada | sí, con estructura |

La fórmula `emb(v1) + ε·t̂` dice **dónde se guarda** la entrada nueva, pero **no dice qué contenido
lleva** ni **cómo se lee**. Sin eso fijado, P1 —la predicción estrella— admite lecturas distintas que
darían números distintos, y elegir después de ver los datos sería exactamente lo que este régimen
existe para evitar.

Nada de lo que sigue es invención: todo se toma de la implementación ya escrita en `exp_gemacion.py`
(R13) y de los resultados R1–R8. Esta enmienda **transcribe** esas decisiones al dominio de la tarea
de hechos y las congela.

---

## 2. Lo que se fija

### 2.1 Dirección y contenido de cada entrada

Una entrada del archivo es un par **(dirección, contenido)**. La dirección es contra lo que se compara
la consulta; el contenido es lo que se devuelve.

| condición | al escribir la revisión | dirección de la entrada nueva | contenido |
|---|---|---|---|
| `sin` | no hay archivo | — | — |
| `sobrescritura` | **reemplaza** la entrada de v1 | `emb(v2)` | valor de **v2** |
| `duplicados` | **agrega** entrada independiente | `emb(v2)` | valor de **v2** |
| `gemacion` | **agrega** entrada anclada | `normalizar(emb(v1) + ε·t̂)` | valor de **v2** |

**La decisión clave, dicha explícitamente:** en `gemacion` la entrada nueva lleva el **contenido de
v2 en una dirección anclada a la posición de v1**. No al revés. Es exactamente lo que hace
`exp_gemacion.py:106-108` (`nueva = self.dirs[j] + EPS_GEM * tangente(...)`, y `self.tok.append(token)`
con el token **nuevo**), y es lo que le da sentido al mecanismo: *la cercanía codifica la correlación*
— la versión nueva vive al lado de la vieja porque son el mismo recuerdo, y lo que cambia es qué dice.

La alternativa —dirección `emb(v2)` atraída hacia `emb(v1)`— **queda descartada** y no se prueba: no
es la gemación de R13 y probar las dos y elegir sería doble oportunidad.

### 2.2 El eje `t̂`

**Por recuerdo, no global**, y **aleatorio, no un campo determinista de la posición**. Esto es R4, que
midió las tres opciones: el eje por recuerdo mantiene M2 = 1,000 incluso a δ = 5 y desacopla
identificar el ítem de recuperar su versión; el eje global acopla las dos métricas y tiene techo duro;
y el eje como campo determinista **pierde** (0,811 vs 0,996) porque el eje rota mientras el recuerdo se
desplaza.

Operacionalmente, igual que `exp_gemacion.py:113-117`: al crear la entrada original se sortea un
vector unitario, se lo hace **tangente** a la dirección (se le quita la componente paralela) y **queda
fijo para ese recuerdo**. Las revisiones sucesivas avanzan sobre ese mismo eje.

### 2.3 ε se mantiene en 0,30, y por qué eso no era obvio

§6 fija ε = 0,30 tomado de R2 sin re-ajustar. **Verificado antes de congelar** que ese valor sigue
teniendo sentido con el encoder nuevo y con textos en minúscula (300 entidades, `nomic-embed-text`):

| magnitud | valor |
|---|---|
| distancia euclídea media entre `emb(v1)` y `emb(v2)` | **0,5824** (rango 0,448–0,695) |
| ε de gemación | **0,30** |
| distancia media a **otras** entidades | **0,9812** |

Esto era un riesgo real de la tarea: como v1 y v2 son textos casi idénticos (misma plantilla, distinto
valor), `duplicados` podría haber obtenido la cercanía **gratis** y entonces P1 no distinguiría nada.
No ocurre: **gemación acerca 1,94×** respecto de la distancia natural, y las dos configuraciones quedan
muy por debajo de la distancia a otras entidades, así que el clúster es separable en ambas.

**ε no se toca.** Si se probara otro valor, se reporta como exploratorio y aparte (§7 del prereg).

### 2.4 Lectura

Idéntica a `exp_gemacion.py:119-131`, con los parámetros del prereg (**k = 5**, umbral de similitud
sin cambios):

1. Se recuperan los **top-k** vecinos de la consulta por coseno.
2. **VIGENTE** = del clúster recuperado, el de **mayor** contador de revisión.
3. **ANTERIOR** = el **penúltimo** por ese contador. Si el clúster tiene un solo elemento, ANTERIOR no
   existe y cuenta como fallo.
4. **COBERTURA** = ambas versiones del recuerdo consultado aparecen entre los top-k.

El contador de revisión es un **metadato entero**, no geometría. Eso es el dictamen de R1+R4:
**geometría para agrupar, metadato para ordenar**. La geometría, sola, recupera la versión más vieja
(M1 ≈ 0) — no ordena.

**Esto acota lo que P1 puede afirmar, y conviene decirlo ahora:** P1 mide si la geometría de la
gemación mejora la **cobertura del clúster**, no si resuelve el orden por sí sola. El orden lo resuelve
el metadato en las cuatro condiciones por igual, así que no puede favorecer a ninguna.

### 2.5 Qué se le pide al archivo en cada condición

Para que `sobrescritura` sea un control válido de P2 (≈ azar en ANTERIOR), su archivo **no guarda** la
entrada de v1: la reemplaza. Es la definición de §3 y no se altera.

`sin` no tiene archivo: responde con lo que el sustrato devuelva sin ningún índice. Es el piso.

---

## 3. Lo que NO cambia

- **Ningún umbral ni margen.** P1 y P3 siguen con margen absoluto 0,02; k = 5; ε = 0,30; 10 semillas;
  IC por t de Student con 9 gl.
- **Las predicciones P1–P4 se leen igual** que en el prereg original.
- **El compromiso de §7 sigue en pie:** no se cambia ε, k ni el umbral después de ver resultados; y si
  P1 cae, se reporta como negativo.
- **D1 y E1 siguen vigentes.**

## 4. Lo que sí cambia respecto del intento anterior

Los embeddings se regeneran **en minúscula** y **en local**. Causa: `nomic-embed-text` servido por
Ollama colapsa a un único vector todo token capitalizado
(`HALLAZGO_TOKENIZADOR_20260810.md`), lo que produjo 75 vectores únicos entre 3000 e hizo que
`emb(v1)` y `emb(v2)` fueran **idénticos bit a bit**. Con el texto en minúscula la compuerta abre
(400/400 vectores únicos, top-1 0,975, rango mediano 0).

`compuerta_encoder.py` se corre **antes** de generar los datos definitivos y **aborta** si falla
cualquiera de sus cuatro chequeos.

## 5. Registro

Enmienda congelada con hash y anclada por push antes de generar dato alguno, siguiendo el mismo
régimen que el prereg original y que E-006 en telar-ligamento.
