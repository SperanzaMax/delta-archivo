# INFORME · dos detectores y la bandera — la vía se CIERRA, y deja dos cosas

Evalúa `PREREG_DOS_DETECTORES.md` (SHA `91494aa0`, congelado 12:25 UTC antes de implementar).
Desviaciones en `DESVIACIONES_DOS_DETECTORES.md` (cuatro, dos de ellas por errores propios).
Seis unidades a 26000 pasos, n = 6000 de ajuste + 6000 de prueba, semillas de generación distintas.
Todo en CPU, sin tocar el pool.

---

## 1. Veredicto

| predicción | criterio | resultado | |
|---|---|---|---|
| **D-0** bloqueante | reproduce el instrumento oficial | identidad **exacta** en 4 de 4 cifras | **PASA** |
| **D-1** principal | componer supera al único por ≥ 0,05 en ≥ 2/3 | −0,0002 · −0,0101 · −0,0028 | **NO CUMPLE 0/3** |
| **D-2** la bandera | foco ≥ 0,70 **y** gana a `pos_q` por ≥ 0,05 en ≥ 2/3 | −0,0498 · −0,0063 · −0,1210 | **NO CUMPLE 0/3** |
| **D-3** nulo, tenía que fallar | 0,50 ± 0,03 | 0,4885 a 0,5071 en las **seis** | **OK** |
| **D-4** réplica en `lat2` | sin criterio | D-2 **no evaluable**, ver §4 | reportado |

> **§5 del prereg, comprometido por adelantado: «Si D-1 falla y D-2 falla, la línea de la bandera se
> cierra. No se prueba una segunda forma de emitir la señal, ni una tercera featura, ni otro
> clasificador.»** Es la celda que salió. **LA BANDERA SE CIERRA.**

---

## 2. D-2 · La bandera falla, y falla en la dirección contraria

La idea era que la señal vive en el foco de lectura y se diluye antes de llegar a donde se decide.
La premisa **espacial** es correcta y quedó medida: el foco cae en la posición 2,54-4,97 y la
decisión en la 7,10, y **coinciden en el 0,0-0,2 % de los casos**. Son lugares distintos casi
siempre.

Lo que resultó falso es que eso importe.

| unidad | foco | `pos_q` | Δ |
|---|---:|---:|---:|
| `p3_s0` | 0,6287 | 0,6785 | −0,0498 |
| `p3_s1` | 0,7598 | 0,7661 | −0,0063 |
| `p3_s2` | 0,6496 | 0,7706 | **−0,1210** |

**`pos_q` gana en las tres, y además es más estable** (0,679-0,771, contra 0,629-0,760 del foco).
El estado recurrente llega al punto de decisión habiendo **conservado** lo que hacía falta, no
habiéndolo perdido. Una bandera transportaría la señal desde donde está *peor* medida.

Esto no contradice `INFORME_FOCO_LECTURA_20260816.md` —la **distribución de lectura** en `pos_q` sí
es casi uniforme, entropía 1,71-1,77 contra un techo de 1,79— y ahí está la precisión que corrige la
intuición: **la lectura se difumina, pero el estado que la integró no.** La evidencia no vive en la
distribución de atención sino en lo que el modelo hizo con ella.

## 3. D-1 · Componer no ayuda, y el primer número era un artefacto

El D-1 original dio −0,3174 y **no era un resultado**: el blanco estaba condicionado a la decisión
del propio detector que se quería evaluar (`DESVIACIONES` D-D3). Con el blanco limpio —«si el modelo
contestara un valor, ¿estaría mal?»— el efecto desaparece casi entero:

| | `p3_s0` | `p3_s1` | `p3_s2` |
|---|---:|---:|---:|
| único | 0,9696 | 0,7986 | 0,8590 |
| compuesto | 0,9693 | 0,7885 | 0,8562 |
| **Δ** | **−0,0002** | **−0,0101** | **−0,0028** |

Componer dos detectores especializados **no ayuda ni perjudica**. La hipótesis de que el detector
único fallaba *por* mezclar dos problemas queda **refutada**: una sola sonda con acceso a las mismas
featuras alcanza lo mismo.

**Lo que hay que decir de la contaminación, porque es la lección y no el número:** el artefacto valía
−0,32, o sea **treinta veces** el efecto real. Y no se cazó por intuición sino por una contradicción
interna — la misma sonda predecía ausencia con AUC 0,8403 y error con 0,3453, invertida, cuando la
tasa base exigía que fuera > 0,5.

## 4. Lo que la réplica en `lat2` mostró, y no estaba buscado

D-2 quedó **no evaluable** en las tres unidades `lat2`, y el motivo es el resultado:

| | `v3_s0` | `v3_s1` | `v3_s2` |
|---|---:|---:|---:|
| `err_identidad` | **0,0000** | **0,0000** | **0,0000** |

Es una **réplica independiente** del cierre de `lat2` del 25-ago: otro instrumento, otras semillas de
generación, n = 6000, y el mismo cero exacto. El resultado se sostiene fuera de su propio
experimento, que es más de lo que se le había pedido.

Y reordena el problema: **la mala atribución —lo que este proyecto llamó alucinación durante
semanas— ya está resuelta.** Lo que queda es `invento`, contestar cuando no hay nada, que es otro
fallo con otro mecanismo.

## 5. Los dos hallazgos que la corrida deja en pie

### 5.1 · La cabeza está optimizada para el blanco equivocado

Evaluando **sobre lo que un detector tiene que anticipar** —«¿me voy a equivocar si contesto?»—:

| detector | `p3_s0` | `p3_s1` | `p3_s2` |
|---|---:|---:|---:|
| la cabeza **que existe hoy** | 0,9598 | 0,7068 | 0,8105 |
| sonda lineal sobre **el mismo estado que ella lee** | 0,9696 | 0,7986 | 0,8590 |
| **cabeza + confianza de salida (4 números)** | **0,9761** | **0,8155** | **0,8722** |

**La información está en el estado que la cabeza ya lee, y la cabeza no la usa.** No es techo de
capacidad: una sonda lineal sobre el mismo vector la recupera. La cabeza aprende «¿hay respuesta?» y
se la juzga por «¿te equivocaste?», que no es lo mismo.

**Y hay una mejora gratis, sin reentrenar nada:** sumarle la confianza de salida da **+0,0163 ·
+0,1087 · +0,0617**. La ganancia es grande justo donde la cabeza es mala (0,71 → 0,82) y chica donde
ya está en el techo (0,96 → 0,98), que es el patrón que uno querría.

### 5.2 · La mala atribución es más detectable que la ausencia, y es estable entre semillas

| | `p3_s0` | `p3_s1` | `p3_s2` | rango |
|---|---:|---:|---:|---|
| detectar **ausencia** | 0,9776 | 0,8403 | 0,8992 | 0,137 |
| detectar **mala atribución** (sólo con respuesta) | 0,8906 | 0,8786 | 0,8812 | **0,012** |

El segundo renglón varía **once veces menos** entre semillas. Mientras la calidad del modelo cambia
mucho de semilla a semilla —`cabeza_sola` va de 0,71 a 0,96—, **la señal de mala atribución está
disponible en la misma medida siempre**. Es observacional y no estaba pre-registrado, pero es la
observación más limpia de la corrida.

## 6. Lo que este experimento no contesta

- **Todo es supervisado.** No habilita «el modelo sabe cuándo no sabe».
- **Las sondas son post-hoc.** Que una regresión logística recupere la señal no prueba que una cabeza
  entrenada la alcance: eso es lo que `PREREG_BLANCO_ERROR.md` (SHA `d065838f`) va a medir.
- **Nada sobre escala.** 863.730 parámetros, idioma sintético de 242 tokens.
- **La sonda de D-2 tiene 264 featuras y la de atribución 132.** No se barrió regularización ni se
  probó otro clasificador, a propósito: el prereg lo prohibía y el nulo en 0,50 dice que no hacía
  falta.

## 7. Registro de errores propios en esta corrida

Van juntos para que se vean, que es la única forma de que sirvan.

1. **Elegí como unidad principal una condición que había eliminado el fallo que quería medir**
   (D-D1). Estaba en dos documentos míos que no crucé.
2. **Definí el blanco de D-1 condicionado a la decisión del detector** (D-D3). Valía 30× el efecto
   real.
3. **Puse un criterio de sanidad más fino que el ruido de muestreo** (D-D4): pedía ±0,02 contra un
   número medido en otra muestra, con error estándar combinado de 0,014.

Las tres se detectaron **antes** de reportar, dos de ellas por controles que podían fallar y una por
una contradicción aritmética. Ninguna la detectó la intuición.
