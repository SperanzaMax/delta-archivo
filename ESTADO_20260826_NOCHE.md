# ESTADO · 26 de agosto, cierre

Para retomar mañana sin releer nada más que este archivo.

---

## 1. Lo que se cerró hoy

### El trípode SE ESCRIBIÓ y el paper está listo

`preprint/tripode/` — **«Where Abstention Lives»**, 10 páginas, inglés como manuscrito principal y
castellano completo como suplemento. Commit `ef79757`. **No corrió nada**: todo estaba medido.

`METADATOS_ENVIO.md` tiene el abstract en texto plano listo para pegar, las declaraciones y una
tabla de trazabilidad de cada número contra su informe.

**Decisión de Maxi: se publica MAÑANA, con lo medido el 26 adentro.** El motivo, en sus términos, es
que un preprint tiene DOI y es permanente, y que A5 puede tocar este paper y no el siguiente.

**Para publicarlo hacen falta dos cosas suyas:** la extensión de Chrome conectada, y que él inicie
sesión en Research Square (el login no lo hago yo). Su sesión estaba en **otra ventana** de Chrome,
no en la que abrí.

### La bandera se CERRÓ por su criterio

`PREREG_DOS_DETECTORES.md` (SHA `91494aa0`) · `INFORME_DOS_DETECTORES_20260826.md`.
D-1 no cumple 0/3, D-2 no cumple 0/3, el nulo se comporta en las **seis** unidades.

**Lo que queda en pie y sirve:** la cabeza está optimizada para el blanco equivocado
(0,9598/0,7068/0,8105 sobre «¿me voy a equivocar?»), y **sumarle la confianza de salida da +0,016 /
+0,109 / +0,062 sin reentrenar nada**.

### Calibrar y ensamble, los dos contrastes que faltaban

`INFORME_CALIBRA_ENSAMBLE_20260826.md`, exploratorio y declarado como tal.
**Calibrar sube `nose` +0,035 / +0,056 / +0,039 gratis** y captura ~75 % del oráculo. El ensamble
gana donde el modelo es malo y pierde donde es bueno.

---

## 2. Lo que está CORRIENDO, y cómo retomarlo

**A5 — `--blanco error`.** `PREREG_BLANCO_ERROR.md` (SHA `d065838f`) + enmienda (`07191f7f`) +
`DESVIACIONES_BLANCO_ERROR.md`.

La cabeza deja de aprender «¿hay respuesta?» y pasa a aprender **«¿me voy a equivocar si contesto?»**.
Control pareado: `p3_s0/s1/s2`, ya corridas, idénticas salvo el blanco.

**Comando exacto para relanzar** (desde `micro_lm/`):

```bash
PREFIJO=b P_NOSE=0.4 ABST=cabeza DONDE=pre BLANCO=error SEMBRAR=0 REINIT=0 HORIZONTE=26000 \
  nohup ./rotar_abst3.sh 3:0,3:1,3:2 26000 2000 250 > rot_b3_20260826.log 2>&1 &
```

`SEMBRAR=0` **no es opcional**: `p3_*` y `v3_*` nunca se sembraron, y sembrar rompe el pareo además
de chocar con la guarda de horizonte (ver `DESVIACIONES_BLANCO_ERROR.md` D-B1).

### Lo que va midiendo, y hay que leerlo con cuidado

`b3_s0` contra su control al mismo paso, **eval interna del tramo (512 muestras): sirve para la
tendencia, no para el nivel**.

| paso | `vigente` b3 | `vigente` p3 | `nose` b3 | `nose` p3 |
|---:|---:|---:|---:|---:|
| 3000 | 0,0222 | 0,6356 | 0,9894 | 0,4182 |
| 4000 | 0,5869 | 0,6229 | 0,6445 | 0,5011 |
| 5000 | 0,8232 | 0,6245 | 0,8854 | 0,5369 |
| 6000 | 0,8824 | 0,6106 | 0,8704 | 0,6098 |

Y la eval en vivo cerca de los 8000 daba `vigente` 0,9688 · `anterior` 0,8589 · `nose` 0,9080 ·
`falsa_abst` 0,0221 — o sea **cerca de lo que al control le costó 26000 pasos** (0,9705 / 0,9119 /
0,0082), con la falsa abstención algo más alta.

### La fase transitoria, que es propiedad de la condición y no un accidente

**Las dos semillas se abstienen del 100 % durante ~3000 pasos y después aflojan solas.** `b3_s0`
empezó a salir a 3000 (`vigente` 0,0222) y `b3_s1` en el mismo punto (0,0410).

El mecanismo, verificado con `diag_b3.py`: el blanco sale del **argmax** y no de la cabeza, así que
cuando el argmax mejora, el blanco baja y la cabeza deja de abstenerse. **El circuito está abierto
por diseño.** A 2000 pasos σ(media del logit) era 0,7342 contra tasa base 0,7572 (pegado al prior);
a 4000 pasaba a 0,4845 contra 0,5280, con el desvío del logit subiendo de 0,2780 a 0,7032.

> **Si mañana una semilla se abstiene del 100 %, NO es colapso: hay que dejarla pasar los 3000
> pasos.** Casi la doy por muerta hoy a las 14:30 con el diagnóstico completo en la mano, y lo único
> que lo impidió fue la regla de no leer un negativo sin barrido de presupuesto.

### La métrica que decide

**NO es la compuerta histórica.** El §4 del prereg lo fija por adelantado: con blanco `error` la
cabeza se activa también en preguntas que sí tienen respuesta pero que el modelo erraría, y
`falsa_abst` cuenta eso como falsa abstención cuando es la abstención **correcta**. Usar la compuerta
vieja garantizaría un «fallo» que no diría nada.

**La métrica es el SER a COBERTURA IGUALADA**, en 0,60 · 0,70 · 0,80.

---

## 3. Dos bugs de infraestructura arreglados hoy, y los dos los cazó una guarda

1. **Guarda de horizonte** (ya existía). Abortó el primer lanzamiento y el diagnóstico **corrigió el
   diseño**: descubrió que `p3`/`v3` nunca se sembraron.
2. **Sesión de Colab reusada entre unidades** (`tramo_abst.sh`, arreglado). `/content/ck.pkl`
   sobrevivía en la VM y `b3_s1` cargaba los pesos de `b3_s0`. Sólo pasa con `SEMBRAR=0`. La guarda
   de identidad de semilla lo cazó; **sin ella la campaña habría sido basura en silencio**.

Más la guarda de `blanco` que escribí hoy y tuve que arreglar yo mismo antes de lanzar: comparaba
contra un default y abortaba en el checkpoint sembrado, que no trae la clave.

---

## 4. Lo primero que hay que hacer mañana

1. **Ver dónde quedó A5** y relanzar con el comando de arriba.
2. **Publicar el preprint** — necesita la extensión de Chrome y el login de Maxi.
3. Cuando A5 cierre, medir **SER a cobertura igualada** contra `p3_*` y escribir el informe.

**Y una cosa que salió hoy y vale más que todo lo anterior junto:** el correo institucional destraba
**tres** trámites a la vez (OpenReview → TMLR/ARR/TACL, arXiv, y que su perfil de Scholar sea
buscable). La nota al Rector está redactada desde el 19-ago y sin enviar.

**Perfil de Google Académico creado hoy:** <https://scholar.google.com/citations?user=36xqH_8AAAAJ>
Y su preprint de Research Square **ya está indexado en Scholar**.
