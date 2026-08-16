# CONSULTA · cómo hacemos que el micro-LM diga «no sé»

**2026-08-16** · para revisión externa (Fable 5), con el repo `delta-archivo` delante.
Autocontenido a propósito: los números están acá, no hace falta correr nada para opinar.

---

## 0. Qué es esto en una página

Estamos entrenando **desde cero** el modelo más chico que pueda sostener esta frase:

> *«este modelo no olvidó lo que le dije, supo cuál corrección regía, y cuando no lo sabía lo dijo».*

Es el criterio de éxito que puso Maxi, textual: *«voy a terminar de creer que esto funciona cuando
creemos un modelo de cero con esto incorporado en su ADN, no algo adosado al LLM».* De ahí que no
haya ningún LLM preentrenado en el pipeline y que el archivo de memoria esté **dentro** de la red,
co-entrenado, no adosado como RAG externo.

**El modelo:** 863 730 parámetros · 3,5 MB · idioma cerrado de 242 tokens (100 números, 60 nombres,
30 entidades, 12 relaciones, funcionales y 6 de control, entre ellos `NOSE`) · delta rule + archivo
persistente co-entrenado · JAX · entrena en una T4 de Colab a 0,22 s/paso, 12 000 pasos por corrida.

**La tarea**, en el formato exacto que ve el modelo:

```
sesión 1    USUARIO  el director de norte es ana
sesión 2    USUARIO  no , es beto              ← corrección elíptica: no nombra la entidad
sesión 7    USUARIO  quien dirige norte ?
            MODELO   beto
```

La respuesta es **un solo token** (`beto`, `ana` o `NOSE`) para que la métrica sea exacta y
determinista, sin juez LLM ni parser interpretativo — lección cara: el 12-ago, 10 de 11
«abstenciones» resultaron ser el modelo contestando bien otra cosa y el parser anotándolo como
rechazo.

**Cuatro niveles**, en orden de dificultad: N1 plantilla fija · N2 paráfrasis (cuatro formas del
mismo hecho) · **N3 corrección elíptica** · N4 multi-sesión con estado reseteado entre sesiones y el
archivo persistiendo.

**Mapa del repo** (todo cuelga de `micro_lm/`):

| archivo | qué es |
|---|---|
| `idioma.py` | generador del idioma cerrado, episodios, consultas y preguntas sin respuesta |
| `datos.py` | armado de lotes, sesiones, cortes, `tipo` de consulta, meta para SER |
| `modelo.py` | delta rule + archivo co-entrenado |
| `entrenar.py` | loop, `evaluar()`, checkpoints continuables por tramos |
| `ser.py` | SER desagregado por tipo de error (**la métrica central**) |
| `mitigar.py` | umbral de confianza sobre checkpoints ya entrenados |
| `test_metricas_nose.py` | 10 comprobaciones de las métricas contra 4 modelos falsos |
| `lanzar_nose.sh` / `worker_cola.sh` / `tramo_colab.sh` | campaña distribuida en 13 cuentas de Colab |
| `../DISENO_MICRO_LM.md` | el diseño, con §7 «lo que puede salir mal» escrito antes |
| `../INFORME_SER_20260815.md`, `../NOTA_ABSTENCION_20260815.md`, `../INFORME_MITIGACION_20260815.md` | los tres informes que fundan esta consulta |

---

## 1. Por qué el `NOSE` es el problema central y no un detalle

Un LLM no distingue entre **recuperar** e **inventar**: para él son la misma operación. Cuando la
información está, lo que sigue coincide con la verdad; cuando no está, el proceso no se detiene ni
avisa — sigue produciendo lo que mejor encaja. No hay señal de «no sé» porque **no hay diferencia
mecánica entre los dos casos**.

La tesis del proyecto es que el error silencioso es de otra categoría que el error avisado:
**un error avisado cuesta una respuesta; uno silencioso cuesta la confianza en todas las demás.**
Por eso la métrica principal no es el acierto sino el **SER** (*silent error rate*): errores
contestados **con seguridad** sobre el total. Si el modelo se abstiene, no cuenta como error
silencioso.

---

## 2. Lo que está MEDIDO (2026-08-15)

### 2.1 El error no es de versión, es de identidad — y el modelo nunca fabrica

`ser.py` sobre checkpoints a 12 000 pasos, idioma v2:

| nivel | acierto | SER | err_versión | err_identidad | err_fuera |
|---|---:|---:|---:|---:|---:|
| N1 plantilla fija | 0,9980 | 0,0020 | 0,0020 | 0,0000 | 0,0000 |
| N2 paráfrasis | 1,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| N3 corrección elíptica | 0,7754 | 0,2246 | 0,0020 | **0,2227** | 0,0000 |
| N4 multi-sesión | 0,7598 | 0,2402 | 0,0078 | **0,2324** | 0,0000 |

Tres lecturas:

1. **El versionado está resuelto** (`err_versión ≤ 0,0078`). Es el problema que la línea perseguía
   desde el principio: un índice geométrico «agrupa perfecto pero no ordena», y el archivo
   co-entrenado reproducía el mismo fallo (0,4576 = azar entre la vieja y la nueva) hasta que un
   **sello de orden** lo levantó a 0,9956. Acá aparece cerrado punta a punta.
   *Control obligatorio, porque el reparto no significa nada sin él:* con pocos valores en juego,
   «el valor de otra entidad» se acierta por azar más seguido que «otra versión del mismo hecho».
   Esperado por azar: versión 0,0741 / identidad 0,9259. Observado: 0,0279 / 0,9721. Los errores de
   versión son **2,7× menos que por azar** → no es artefacto del conteo.

2. **Lo que rompe es identificar la entidad, y aparece exactamente en N3**, el nivel de la
   corrección elíptica. La corrección no dice de quién habla, y ahí el modelo se equivoca de dueño.
   La falla cae donde el mecanismo predice.

3. **`err_fuera = 0,0000` en los cuatro niveles: el modelo nunca inventa contenido.** Toda respuesta
   errada es un valor **real del archivo puesto en la entidad equivocada**. La alucinación acá no es
   fabricar un dato sino **atribuir mal uno verdadero** — más difícil de detectar desde afuera,
   porque cualquier verificación de tipo «¿este dato existe?» la da por buena.

Contexto de contraste: el mismo caso elíptico con un índice **no paramétrico** sobre encoder
congelado da **0,0000 exacto en 10/10 semillas**, o sea pierde el 100 % de las correcciones en
silencio (está publicado: DOI 10.21203/rs.3.rs-10669947/v1). El micro-LM da 0,7754 sobre eso mismo.

### 2.2 El hallazgo incómodo: la abstención nunca se midió

**Todas las corridas del proyecto —ocho días de campañas— usaron `p_nose = 0,0`.** O sea: ninguna
pregunta carecía de respuesta en el archivo. Con eso, `NOSE` **no era una opción que el modelo
pudiera tomar**, la métrica de abstención sale `NaN`, y el «abstención = 0,0000» que veníamos
reportando no era un resultado sino una consecuencia del generador.

Verificado antes de gastar GPU (init + `evaluar`, sin entrenar): con `p_nose = 0,2` las métricas
`nose`, `nose_ent`, `nose_rel` se computan; con `0,0` las tres son NaN.

### 2.3 Los dos atajos de la abstención, medidos sobre el generador

| `p_nose` | % preguntas sin respuesta | acierto de **no abstenerse nunca** | acierto de **abstenerse siempre** |
|---:|---:|---:|---:|
| 0,1 | 0,1062 | 0,8938 | 0,1062 |
| 0,2 | 0,2047 | **0,7953** | 0,2047 |
| 0,3 | 0,2898 | 0,7102 | 0,2898 |
| 0,4 | 0,4094 | 0,5906 | **0,4094** |

Con `p_nose = 0,2`, **no abstenerse nunca vale 0,7953 — más que los 0,7598 de nuestra mejor corrida
de N4**. El gradiente no tendría ningún motivo para aprender a abstenerse: la compuerta habría
fallado por diseño nuestro, no por incapacidad del modelo. Por eso la campaña se fijó en 0,4, donde
el atajo cae a 0,5906 y deja de dominar.

En el otro extremo aparece el atajo simétrico. Probado en un proxy chico (d=32, 2 capas, 45 298
params, 3000 pasos):

| régimen | resultado a 3000 pasos | lectura |
|---|---|---|
| `p_nose = 0,0` (control) | vigente **0,1296**, en subida | aprende, lento, lejos de saturar |
| `p_nose = 0,2` | vigente 0,0030 · falsa_abst 0,9953 | colapsa a abstenerse de todo |
| `p_nose = 0,4` | vigente 0,0000 · falsa_abst 1,0000 | colapsa más fuerte |

**El control es lo que hace legible el resultado:** sin él, el colapso se leería como incapacidad del
proxy. Con él se ve que **no es incapacidad — es que abstenerse paga más que un mecanismo a medio
aprender**. Con el mecanismo rindiendo 0,13, decir `NOSE` siempre rinde 0,41. Y `NOSE` es **un**
token, mientras que acertar exige elegir entre cien: el atajo es además el más fácil de encontrar al
arrancar.

Es la misma estructura del **atajo de la recencia** que ya nos mordió antes: con turnos móviles, dos
de tres semillas convergieron a «gana el turno más alto», resolvían la versión vigente en 0,96 y
fallaban la anterior *por debajo del azar*. El gradiente prefiere la solución barata **mientras la
cara no rinda todavía**.

### 2.4 El currículum, probado y mal probado

Se reanudó el control (que venía de 0,1296 con `p_nose = 0`) introduciendo `p_nose = 0,4`: colapsó a
0,0037 en 500 pasos. **Eso no refuta el currículum, lo probó en el punto equivocado** — la premisa
era introducir las preguntas sin respuesta *después* de que el mecanismo sature, y 0,1296 no satura
nada: abstenerse seguía pagando 0,41 contra 0,13. El modelo hizo lo racional.

De ahí el criterio operativo que tenemos escrito:

> **Introducir `NOSE` sólo cuando `vigente` supere la tasa de preguntas sin respuesta**
> (0,41 si `p_nose = 0,4`; 0,20 si 0,2).

Y sale gratis de implementar: la guarda de identidad del checkpoint compara
`nivel, semilla, lr, idioma, d, capas` y **no** `p_nose`, así que una corrida se reanuda cambiando de
régimen; el `p_nose` queda registrado en cada evaluación de la historia para que un salto de métrica
no se lea como aprendizaje.

*Límite declarado:* todo el barrido de régimen es un proxy de 45 298 params / 3000 pasos contra
863 730 / 12 000 reales. **La aritmética de los atajos no depende del proxy; la dinámica sí.**

### 2.5 Mitigación sin reentrenar: funciona para media enfermedad

Antes de gastar GPU probamos lo barato: **el modelo que ya tenemos, ¿sabe cuándo está por
equivocarse?** Tres señales de la distribución de salida (probabilidad del token elegido, margen
1º-2º, entropía):

| checkpoint | AUC prob | AUC margen | AUC entropía |
|---|---:|---:|---:|
| n4_s0 (N4) | **0,8631** | 0,8626 | 0,8602 |
| n3_s2 (N3) | 0,8688 | **0,8688** | 0,8623 |

Con umbral **calibrado en una mitad y medido en la otra** (elegirlo sobre el mismo conjunto es
oráculo), y contra el piso de **abstenerse al azar** en igual proporción:

| cobertura | acierto | SER | SER evitado | vs azar |
|---:|---:|---:|---:|---:|
| 1,00 | 0,7704 | 0,2287 | 1,2 % | 1,01× |
| 0,90 | 0,8077 | 0,1727 | 25,4 % | 1,20× |
| **0,78** | **0,8624** | **0,1073** | **53,6 %** | **1,68×** |
| 0,51 | 0,9844 | 0,0080 | 96,5 % | 14,81× |

Contestar el 78 % y callarse el 22 % **elimina la mitad de los errores silenciosos sin tocar el
modelo**, y apaga el 48 % de los errores de identidad.

**Pero se quiebra justo en el caso central.** Los checkpoints se entrenaron con `p_nose = 0`, nunca
vieron una pregunta sin respuesta; evaluados con `p_nose = 0,4`:

| régimen de evaluación | AUC | invento apagado (cob. 0,80) | vs azar |
|---|---:|---:|---:|
| sin preguntas sin respuesta | 0,8631 | — | 1,68× |
| **con preguntas sin respuesta** | **0,7397** | **28,8 %** | **1,16×** |

**Confía casi igual inventando que acertando.** Mecánicamente es esperable: nada en su entrenamiento
le pidió distinguir «está y es X» de «no está», así que **la ausencia no tiene representación
propia**; el lector devuelve su mejor candidato y el candidato es igual de confiable en los dos
casos.

Consecuencias que ya tomamos: (a) la compuerta de abstención entrenada sigue siendo necesaria;
(b) **ya hay baseline: la campaña tiene que superar el 28,8 % de invento apagado que sale gratis**
—antes no teníamos con qué comparar y cualquier número habría parecido bueno—; (c) umbral y
entrenamiento son complementarios, hay que medirlos juntos sobre un checkpoint con `p_nose > 0`.

---

## 3. El instrumento (esto sí está cerrado y probado)

**Cinco categorías de respuesta**, en `ser.py:clasificar()`:

| categoría | cuándo |
|---|---|
| `acierto` | contesta el token correcto |
| `err_version` | contesta otra versión **del hecho preguntado** (falla del orden temporal) |
| `err_identidad` | contesta el valor de **otra entidad** (falla del direccionamiento) |
| `err_fuera` | ni del hecho ni de los otros: no recuperó nada del archivo |
| `invento` | **no había respuesta y contestó igual** — la alucinación pura |
| `abstencion` | había respuesta y dijo `NOSE` — el costo de saber abstenerse |
| `acierto_nose` | no había respuesta y dijo `NOSE` |

`invento` no existía en el código hasta el 15-ago: con `p_nose = 0` el caso no podía darse, y esos
errores caían en `err_identidad` sin que nadie lo notara.

**Dos tipos de pregunta sin respuesta**, a propósito, porque miden cosas distintas:
- `nose_ent` — la entidad nunca se nombró en el episodio → basta con no encontrarla en el archivo;
- `nose_rel` — **la entidad sí aparece, pero con otra relación** → hay que encontrar la entidad y
  además verificar que la relación pedida no está. Es el caso difícil y el que más se parece a la
  alucinación real.

**Denominadores separados**, porque las dos caras de la abstención no se miden sobre el mismo
universo: `falsa_abst` sobre las preguntas **con** respuesta, `invento` y `nose` sobre las **sin**.

**Validación del instrumento** (`test_metricas_nose.py`): 4 modelos falsos (oráculo, nunca abstiene,
siempre abstiene, azar) y **10 comprobaciones que pueden fallar, todas pasan**. Lo clave: un modelo
que dice `NOSE` a todo saca `nose = 1,0000` **y `SER = 0,0000`**, y sólo lo delatan `falsa_abst = 1`
y `acierto = 0` → **ninguna de las dos métricas sirve sola**. Reproducibilidad desde la semilla
verificada en 4 niveles × 2 regímenes, ensuciando a propósito el RNG global de numpy (que era un bug
real del 14-ago).

---

## 4. El plan que tenemos hoy, y su estado

**Campaña `x`** (`lanzar_nose.sh`), aislada de la base por prefijo en checkpoints, claims y salidas:

- `PREFIJO=x`, `P_NOSE=0.4`, unidades `1:0 4:0 4:1 4:2`, 12 000 pasos cada una.
- **`x1_s0` es la COMPUERTA**, en el nivel fácil, con criterio fijado antes de mirar:
  **sigue si `nose ≥ 0,50` y `falsa_abst ≤ 0,10`**. Si los dos son altos, el modelo abstiene de todo
  y la métrica miente. La compuerta pregunta «¿*puede* abstenerse?», no «¿lo hace bajo la proporción
  final?»: si puede a 0,4 se estudia después a 0,2; si no puede a 0,4, tampoco iba a poder a 0,2.
- Después, N4 en tres semillas — el resultado que importa.

**Estado real al 2026-08-16 08:48: la campaña se lanzó la noche del 15 y corrió CERO pasos.** Los 27
workers pasaron la noche en `sin acelerador · espera 5 min`: no hubo una sola GPU en las 13 cuentas
de Colab. Quedaron además **4 claims huérfanos** (`x1_s0`, `x4_s0`, `x4_s1`, `x4_s2`) que hay que
borrar antes de relanzar, porque el reclamo es atómico y ninguna cuenta va a tomar esas unidades.

**Campaña base** (con `p_nose = 0`), 9 de 12 corridas a 12 000 pasos:

| | s0 | s1 | s2 |
|---|---|---|---|
| N1 | 1,0000 | 1,0000 | 1,0000 |
| N2 | 1,0000 | 0,8028 | 0,9977 |
| N3 | *4000: 0,7209* | *4000: 0,7162* | 0,8264 |
| N4 | 0,7578 | 0,7693 | *4000: 0,7152* |

N2 confirma **bimodalidad**: dos semillas saturan y una queda trabada. Regla que adoptamos: **no
reportar la media de un nivel sin sus tres semillas**, porque con una sola no se distingue dificultad
de no-convergencia.

**Recursos:** sólo Colab gratuito, 13 cuentas, T4 cuando hay. Disponibilidad medida: 8 T4 a las
08:10, 5 al mediodía, 3 a la tarde, **0 a la noche**. CPU descartada (4× más lento y el kernel se
traba al 100 %, muriendo el polling y la bajada de checkpoints). Una corrida de 12 000 pasos ≈ 45 min
de T4; el plan completo de la campaña son ~13 h de GPU repartidas en dos días.

---

## 5. Historial de errores propios, porque acota lo que aceptamos como resultado

Siete veces en este programa **un número limpio escondió un artefacto**. Las formas, para que se
entienda qué tipo de crítica nos sirve:

1. **Control vacío** — `m=1 → 1,000` parecía validar el banco; con una sola candidata, acertar no
   requiere leer. *Un control de sanidad tiene que poder fallar.*
2. **Impaciencia** — «el lector no usa el orden» a 4000 pasos (0,1109) era falso: a 12 000 da 0,9870.
   *Un negativo sin barrido de presupuesto no es un negativo.*
3. **Hiperparámetro heredado** — la ventana de lr es estrechísima: 1e-3 → 0,7305 · 3e-3 → 0,0265 ·
   1e-2 → 0,0130. La corrida principal usó el 3e-3 calibrado para otra tarea y dio azar en las tres
   predicciones.
4. **Padding disfrazado de techo** — tres niveles medían un truncamiento del 34 %, no el modelo. Lo
   delató que el nivel más difícil diera 0,988 y el piso 0,67: *cuando el orden de dificultad sale al
   revés, el problema es del instrumento.*
5. **Fuga por el contenido** — dos versiones en la misma secuencia y el estado de la segunda arrastra
   rastro de la primera: el orden se regalaba.
6. **RNG global** — `np.random.choice` hacía irreproducibles corridas que creíamos reproducibles.
7. **Artefacto del propio instrumento** — el 90 % de las «abstenciones» eran el parser fallando.

Nuestra regla actual: pre-registrar la predicción con hash **antes** de ver los datos, y no mandar un
veredicto sin correr su control y buscar activamente la **explicación alternativa** («¿qué otra cosa
produciría este mismo número?»).

---

## 6. LAS PREGUNTAS (todo junto, en una pasada)

Lo que sigue es donde genuinamente no tenemos respuesta. Contestá lo que puedas **de una vez** (una
sola pasada, no rondas); las precisiones de implementación las resolvemos de este lado.

**Si tenés que repartir esfuerzo, las tres que más nos mueven son P3 (la ausencia como cantidad
interna), P9 (abstenerse sobre la versión, no sólo sobre la existencia) y P11 (qué se nos escapa).**
Las demás son importantes pero acotadas.

**P1 · El óptimo local de la abstención.** El problema está bien caracterizado: `NOSE` es un token
contra cien, y abstenerse rinde más que un mecanismo a medio aprender. Nuestro plan es un currículum
gateado por el criterio «introducir `NOSE` cuando `vigente` > tasa de preguntas sin respuesta».
¿Es la salida correcta, o el currículum sólo mueve el problema? Alternativas que consideramos y no
sabemos ordenar: (a) ponderar la pérdida de `NOSE` a la baja al principio; (b) rampa continua de
`p_nose` en vez de escalón; (c) **una cabeza de abstención separada** en vez de un token que compite
en el mismo softmax; (d) una pérdida auxiliar de «¿está esta clave en el archivo?» sobre el propio
score de recuperación.

**P2 · ¿Abstención aprendida o calibración? Y qué tendría que superar.** Sin entrenar nada, un umbral
de confianza apaga el 28,8 % del invento (1,16× el azar). Con `p_nose > 0` esperamos que el modelo
desarrolle una representación de la ausencia. **¿Qué número haría que valga la pena, y cuál sería el
control honesto?** Nos preocupa la trampa simétrica: que el modelo entrenado con `p_nose = 0,4`
mejore el `nose` simplemente porque abstenerse es frecuente, sin haber aprendido nada sobre la
ausencia.

**P3 · La ausencia como cantidad interna, no como token.** El diagnóstico de fondo del programa es
que un LLM no distingue recuperar de inventar porque **no hay diferencia mecánica** entre los dos
casos. Un archivo co-entrenado sí tiene un lugar natural donde podría haberla: el **score de la
consulta contra las claves archivadas**. ¿Vale la pena un mecanismo explícito de «no-match» (un slot
nulo aprendido, un umbral sobre el score máximo, un logit de ausencia) en vez de esperar que el token
`NOSE` lo aprenda solo? Esto es lo que más nos interesa, porque es lo único que sería **arquitectura
y no entrenamiento**.

**P4 · El régimen de evaluación.** Entrenamos con `p_nose = 0,4` porque a 0,2 el atajo de no
abstenerse nunca domina (0,7953 > 0,7598). Pero 41 % de preguntas sin respuesta no se parece a
ningún uso real. ¿Es legítimo entrenar a 0,4 y **evaluar a 0,2 o 0,1**, declarando el desajuste? ¿O
hay una forma mejor de romper el atajo sin distorsionar la distribución (por ejemplo, costo asimétrico
del error silencioso en la pérdida en vez de más preguntas sin respuesta)?

**P5 · El error de identidad, que es el 93 % de lo que falla.** `err_identidad` = 0,2227 en N3 y
`err_fuera` = 0,0000: el modelo trae un valor real y lo pega al dueño equivocado. No sabemos si la
corrección elíptica se pega al hecho equivocado **al escribir** o si la consulta recupera el hecho
equivocado **al leer**. Tenemos una hipótesis escrita (`NOTA_FOCO.md`: la identidad se **captura al
escribir**, con el foco puesto, y no se reconstruye al leer). **¿Cómo separarías las dos causas con un
experimento barato?**

**P6 · Prioridad, con el presupuesto real.** Colab gratis, 13 cuentas, 0 a 8 T4 según la hora, ~45 min
por corrida de 12 000 pasos. Sobre la mesa: (a) la campaña `x` como está; (b) cerrar las 3 semillas
faltantes de la base para que N3/N4 tengan sus tres; (c) el mecanismo de ausencia de P3; (d) el
experimento de P5. **¿En qué orden, y qué recortarías si sólo entra una?**

**P7 · Encuadre y literatura.** ¿Contra qué nos van a comparar y qué nos falta citar? Sabemos de
selective prediction / umbral de confianza, de FAMA (arXiv 2604.20006, que penaliza el reuso de
memoria invalidada pero **no separa versión de identidad**) y de LongMemEval (que ya tiene el tipo de
pregunta «knowledge updates»). **¿Qué estamos sobre-interpretando?** En particular: ¿es defendible
presentar `err_fuera = 0,0000` como «el modelo nunca fabrica», o es una consecuencia trivial de tener
un vocabulario cerrado de 242 tokens?

**P8 · ¿Nuestra compuerta puede fallar?** `x1_s0` corre en N1 (plantilla fija) y decide si sigue la
campaña. Nos preocupa repetir el error del control vacío: en el nivel fácil, `nose_ent` (la entidad
nunca se nombró) podría resolverse con «no vi ese token en la sesión», **sin consultar el archivo** —
y entonces la compuerta pasaría sin probar nada de lo que nos importa. `nose_rel` (la entidad está,
la relación no) sí exige ir al archivo. **¿La compuerta debería ser sólo `nose_rel`?** ¿O directamente
correrla en N3, aceptando que un fallo se vuelva ambiguo entre «no puede abstenerse» y «no resuelve
la elipsis»?

**P9 · Abstenerse sobre la VERSIÓN, no sólo sobre la existencia.** Hoy `NOSE` significa una sola
cosa: «ese hecho no está en el archivo». Pero hay un tercer estado que este proyecto midió durante
meses y nunca trató como abstención: **el hecho está, hay dos versiones, y no puedo determinar cuál
rige.** Ese era exactamente el modo de falla histórico (0,4576 ≈ azar entre la vieja y la nueva): el
modelo **tiraba una moneda en vez de decir que no sabía**. Un modelo que dijera «sé el hecho, no sé
cuál versión rige» sería un resultado nuestro y no un tema prestado de la literatura de calibración.
**¿Vale la pena partir la abstención en dos (ausencia vs. ambigüedad temporal), o eso diluye el
mensaje y complica una métrica que ya tiene cinco componentes?**

**P10 · La figura de mérito.** Reportamos `acierto`, `SER`, `nose`, `falsa_abst` e `invento`, y está
probado que **ninguna sirve sola** (un modelo que dice `NOSE` a todo saca `nose = 1,0000` y
`SER = 0,0000`). Para un paper hace falta un número principal. ¿Hay una figura escalar defendible y no
gameable —área bajo la curva riesgo-cobertura, o una utilidad con costo explícito por tipo de error—
que puedas recomendar? Y si es una utilidad con costos: **¿cómo se justifica el costo relativo del
error silencioso frente al de la abstención sin que parezca elegido para que dé bien?**

**P11 · Qué se nos escapa.** La pregunta abierta, y la que más valor tuvo históricamente cuando la
hicimos: mirando el conjunto —el diagnóstico, el instrumento, el plan—, **¿cuál es el experimento
obvio que no estamos haciendo?** ¿Y hay algo acá que ya esté resuelto en la literatura y estemos por
gastar dos días de GPU en redescubrir?

**P12 · Escala, que es el flanco por el que nos van a pegar.** El idioma cerrado de 242 tokens es *la
interfaz de la prueba, no el objeto de estudio* — pero un revisor va a decir «juguete». La vara de
Maxi pide que el modelo no olvide «lo que lee o lo que hablamos», o sea texto real. **¿Cuál es el
cambio más chico que subiría la clase de afirmación?** Opciones que vemos: (a) vocabulario abierto
con nombres nunca vistos en entrenamiento; (b) un N5 con texto natural corto; (c) dejar el idioma
cerrado y defenderlo explícitamente como *aislamiento de variables* —que es lo que ningún modelo
grande permite hacer—. ¿Cuál sostiene mejor el resultado?

---

## Anexo · para correr y ver los números

```bash
cd micro_lm
python ser.py ckpts/n4_s0.pkl --n 2000          # SER desagregado
python mitigar.py ckpts/n4_s0.pkl               # curva riesgo-cobertura
python test_metricas_nose.py                    # las 10 comprobaciones del instrumento
python -c "import datos, numpy as np; print(datos.lote(np.random.default_rng(0), 4, nivel=4, p_nose=0.4, con_meta=True)[7])"
```

Los checkpoints (`ckpts/*.pkl`, ~10 MB cada uno) **no están en git**; están en el disco de Maxi.
Todo lo demás sí.
