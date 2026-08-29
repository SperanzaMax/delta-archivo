# ESTADO · 28 de agosto, cierre

Para retomar mañana sin releer nada más que este archivo.

---

## 1. Lo que se cerró hoy

### El preprint del trípode está PUBLICADO
DOI **10.21203/rs.3.rs-10839567/v1**, live desde las 04:27. Enviado ayer 12:38, o sea 16 horas de
screening. Publicado en LinkedIn con el link y la imagen.

**Y ya llegaron tres invitaciones de revistas depredadoras** en las primeras 12 horas
(`academicvision.org`, `asjpublisher.com`, y una desde un Gmail suelto). No se contesta ninguna.

### La calibración post-hoc se CIERRA como vía
`INFORME_CALIBRA_TRANSFERENCIA_20260828.md`, evalúa `PREREG_CALIBRA_TRANSFERENCIA.md` (SHA
`5fdab03d`, congelado antes de escribir el instrumento).

| | criterio | resultado |
|---|---|---|
| **K-0** el nulo | ≤ 1/20 | **0/20 en las seis** ✓ |
| **K-1** principal | ganancia ≥ 0,03 en ≥5/6 | 3/6 ✗ |
| **K-2** transferencia | retiene ≥60 % en ≥4/6 | **0/6** ✗ |
| **K-4** banda de `z*` | reportar | [−0,64, −0,10], fuera de ±0,15 |

**A3 replica 3 de 3** (+0,0409 · +0,0614 · +0,0371 contra +0,0353 · +0,0562 · +0,0391 del 26-ago),
así que el positivo de aquel día **no era** artefacto de la muestra compartida.

**El resultado es el negativo de K-2: el corte NO transfiere.** En 3 de 6 el corte prestado rompe la
falsa abstención (0,4424 · 0,2609 · 0,3784 contra un límite de 0,10). Los `z*` no comparten banda, o
sea que no hay un sesgo común del nivel: hay seis cortes distintos. Esto **acota** el 7/8 del 20-ago,
que eran ocho unidades homogéneas de la familia `c`; con dos familias mezcladas se cae.

**Defecto propio, declarado:** K-1 le pedía ≥0,03 de mejora a `b3_s0` y `b3_s1`, que tenían techo de
+0,0008 y +0,0012. Es el mismo error que el §4 del informe de A5 dejó anotado el 27 **como lección
para el próximo pre-registro**, y el próximo fue éste. Regla de acá en más: **todo umbral de mejora
absoluta va con el margen al techo calculado antes de congelar.**

---

## 2. LO MÁS IMPORTANTE DEL DÍA · la métrica estaba mal, y lo vio Maxi

> «No queremos un modelo callado, queremos un modelo que dé la respuesta correcta o que diga no sé.
> ¿Qué tipo de respuesta es una que no se dice?»

**Dos correcciones que salen de ahí:**

**1. En este modelo no existe el silencio.** `NOSE` es un token del vocabulario, así que abstenerse
es **emitir la respuesta «no sé»**. Una unidad con `falsa_abst = 1,0000` no está muda: está
afirmando «no sé» en el ~60 % de preguntas que **sí** tenían respuesta. Es una respuesta falsa. No
llamarlo «callado».

**2. La métrica `nose` premia la degeneración.** El proyecto la rotula «la mitad que importa», pero
un modelo que contesta NOSE a todo saca `nose = 1,0000` **e** `invento = 0,0000`, perfecto en las
dos. Medido hoy en `b3_s3`, `b3_s6` y `b3_s7`.

**La métrica correcta es la EXACTITUD GLOBAL**, `(acierto + acierto_nose) / n`. Instrumento nuevo en
`micro_lm/exactitud.py`. **El piso trivial es 0,4065**, que es la fracción de preguntas sin respuesta.

| unidad | paso | exactitud | contestó bien | dijo NOSE con razón | dice NOSE |
|---|---:|---:|---:|---:|---:|
| b3_s0 | 26000 | **0,9995** | 0,5935 | 0,4060 | 0,41 |
| b3_s1 | 26000 | **0,9995** | 0,5930 | 0,4065 | 0,41 |
| b3_s2 | 26000 | 0,6485 | 0,3980 | 0,2505 | 0,40 |
| b3_s4 | 18000 | 0,6515 | 0,3950 | 0,2565 | 0,40 |
| b3_s3 | 22000 | **0,4065** | 0,0000 | 0,4065 | **1,00** |
| b3_s6 | 23000 | **0,4065** | 0,0000 | 0,4065 | **1,00** |
| b3_s7 | 12500 | **0,4065** | 0,0000 | 0,4065 | **1,00** |

**Todo criterio de éxito que mire sólo `nose` está mal construido.**

---

## 3. El hallazgo de la mañana, y la campaña que lo está juzgando

`HALLAZGO_PUNTO_PROPIO_20260828.md` (post-hoc, sobre archivos que ya estaban en disco). En su punto
de operación propio, `b3_s0` y `b3_s1` se callan casi perfecto —`nose` 0,9994 y 1,0000, `falsa_abst`
0,0000 y 0,0008— y `b3_s2` no, pero **falla por recuperación y no por abstención** (`vigente` 0,6762,
o sea ni pasa E-0, y le vuelve `err_identidad` a 0,0447).

Se lanzó `PREREG_TASA_REGIMEN.md` (SHA `dc62ecae`) para saber si 2 de 3 es la tasa o fueron dos
semillas con suerte. Seis unidades nuevas `b3_s3`…`b3_s8`, flags idénticos a A5, sólo cambia la
semilla. **T-1 pide ≥3 de 6 en el régimen** (`nose` ≥ 0,99 **y** `falsa_abst` ≤ 0,01).

**Verificado hoy, porque es lo que haría inválida la comparación:** el código de entrenamiento **no
cambió**. Último commit sobre `entrenar.py`/`modelo.py`/`datos.py`/`idioma.py` es del 26-ago 14:36,
cuando se lanzó A5, y el árbol está limpio. Y la config guardada en los checkpoints nuevos difiere de
`b3_s0` **sólo en la semilla y el nombre del archivo de salida**.

### ⚠ Cómo viene, y no viene bien
Tres de seis (`s3`, `s6`, `s7`) están en **abstención total** a 22000, 23000 y 13500 pasos. Las
originales a esta altura ya hablaban. `s4` despertó a los 13000 pero con el perfil de `s2`, la rota.

**T-0 es bloqueante y pide `vigente` ≥ 0,70 en ≥4 de 6.** Las tres degeneradas están en 0,0000. Si
llegan así a 26000, la campaña no se lee y **el hallazgo del §3 se retira**, que es el desenlace que
el pre-registro contempla en T-4.

**Queda margen:** `s4` estuvo en abstención total hasta los 12500 y despertó de golpe.

### La hipótesis que esto sugiere, si se confirma
Una unidad degenerada tiene **detección perfecta** y cero alucinación, porque no contesta nunca. O
sea que **callarse es trivial y lo difícil es lo otro, salir del silencio sin empezar a inventar.**
Si T-1 falla, la pregunta del proyecto está mal planteada y hay que reformularla en esos términos.

---

## 4. Dos bugs de infraestructura, los dos con la misma forma

**1. La tercera unidad de cada rotador nunca corría.** El rotador pide **una sesión de Colab por
vuelta** y corre las unidades en fila dentro de ella. La primera se lleva ~40 min, la segunda otros
~40, y para la tercera Colab ya recicló la sesión: `Session not found` y tramo cerrado en el paso 0.
Los **15 fallos del día fueron exclusivamente de `s5` y `s8`**, las terceras de cada lista, que
perdieron nueve horas. Arreglado dándoles rotador propio y partiendo las listas a dos unidades.

**2. Matar un rotador no alcanza.** Sus `tramo_abst.sh` **sobreviven reparentados** y siguen bajando
checkpoints de la VM vieja, pisando lo que escribe el rotador nuevo. Provocó un retroceso real de
`b3_s3` de 20000 a 19000. Hay que barrer los hijos, y por eso existe `micro_lm/cierre_20260828.sh`.

---

### Pasos exactos al cierre (21:39)

| unidad | paso | falta |
|---|---:|---:|
| b3_s0 · s1 · s2 | 26000 | — (campaña A5, ya juzgada) |
| **b3_s6** | **25000** | 1000 |
| **b3_s3** | **22000** | 4000 |
| **b3_s4** | **19000** | 7000 |
| **b3_s7** | **13500** | 12500 |
| **b3_s8** | **8000** | 18000 |
| **b3_s5** | **6000** | 20000 |

Faltan **63000 pasos** de 156000. A las ~5000/hora medidos hoy son unas **13 horas**. El cuello son
`s5` y `s8`, que perdieron nueve horas por el bug del §4.1.

Cierre ordenado con `cierre_20260828.sh`: rotadores parados, tramos hijos barridos, las dos sesiones
de Colab vivas (`tr2_h_2119` y `tr2_n_2052`) paradas para no gastar cuota, locks limpios, **cero
procesos vivos**. Todos los checkpoints guardan su paso, así que mañana se reanuda sin perder nada.

---

## 5. Lo primero que hay que hacer mañana

1. **Relanzar los cuatro rotadores** para completar la campaña. Comando en el §6.
2. **Cuando lleguen a 26000, juzgar `PREREG_TASA_REGIMEN` con `ser_cobertura.py`** (campo `propio`,
   n=4000, semilla 54321) **y reportar también la exactitud global.** Salga como salga.
3. **Reddit**, si el resultado lo permite. `r/MachineLearning` con flair `[R]` y `r/LocalLLaMA`, sobre
   el **preprint del trípode**, nunca sobre el hallazgo de hoy si quedó retirado. Anticipar en el
   propio post la crítica de escala, que está declarada en el paper.

## 6. Cómo se relanza la campaña

```bash
cd ~/Documentos/Nuevo\ Transformer/delta-archivo/micro_lm
COM="PREFIJO=b P_NOSE=0.4 ABST=cabeza DONDE=pre BLANCO=error HORIZONTE=26000"
env $COM ACEL=tpu LOG_ROTADOR=$PWD/rot_s3s4_0829.log setsid ./rotar_abst3.sh 3:3,3:4 26000 2000 250 A J F D C > rot_s3s4_0829.log 2>&1 &
env $COM ACEL=t4  LOG_ROTADOR=$PWD/rot_s6s7_0829.log setsid ./rotar_abst3.sh 3:6,3:7 26000 2000 250 L K H M I > rot_s6s7_0829.log 2>&1 &
env $COM ACEL=tpu LOG_ROTADOR=$PWD/rot_s5_0829.log   setsid ./rotar_abst3.sh 3:5     26000 2000 250 F E D C   > rot_s5_0829.log 2>&1 &
env $COM ACEL=t4  LOG_ROTADOR=$PWD/rot_s8_0829.log   setsid ./rotar_abst3.sh 3:8     26000 2000 250 M N I G   > rot_s8_0829.log 2>&1 &
```

**Nunca más de dos unidades por rotador**, por el bug del §4.1. Y para parar, usar
`cierre_20260828.sh`, que barre los hijos.

## 7. Otras cosas del día

- **Modelio 5.4.1 instalado** (snap). Se le hizo el diagrama de clases del ejercicio de sueldos de la
  UTN, en `.uml`, `.dot`, `.png`, `.svg` y `.pdf`. El XMI que traía Maxi **no era XML válido** (un `<`
  sin escapar en `List<Empleado>`) más cinco errores de estructura UML.
- **`VOCABULARIO_MICROLM.pdf`**, las 242 palabras del idioma agrupadas. No son 128: ese número es `d`,
  el ancho interno. Hallazgo lateral: **`vale` es la única palabra con dos papeles**, nombre de
  persona y verbo de la relación `clave`.
- Sigue pendiente, y es lo de mayor palanca: **el correo institucional**. Hoy llegó otro aviso de
  Scholar avisando que el perfil no aparece en búsquedas sin él.
