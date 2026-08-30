# ENMIENDA a `PREREG_RECOMPENSA.md` (SHA `f1f7bb66`) · el peso F estaba mal derivado

**2026-08-29, 22:40.** Se escribe **después** de la primera corrida de la Etapa 1 y **antes** de la
segunda. Los datos que la motivan están reportados abajo, sin recortar.

---

## 1. Lo que se midió, y es un fracaso limpio

Etapa 1 con $L=M=0{,}5$, $F=1{,}5$. Ocho unidades de doce llegaron a 3000 pasos (las otras cuatro se
quedaron sin VM, Colab dio 503 en las trece cuentas durante ocho vueltas).

| unidades | `abstencion` | `vigente` |
|---|---:|---:|
| `tk3_s4/s5/s7/s8` (token) | **0,0000** | 0,0000 – 0,0275 |
| `hd3_s3/s6/s7/s8` (cabeza) | **0,0000** | 0,0033 – 0,0085 |

**No se abstienen nunca, ni una vez, en ninguna de las ocho.** Pasaron del extremo mudo
(`abstencion` 1,0000) al extremo locuaz (0,0000).

W-1 pedía «salir del silencio» y eso **se cumple**, pero de la forma que lo vuelve irrelevante. El
modelo no aprendió a decidir, aprendió la política constante del otro lado.

## 2. La causa, y es un error mío en la derivación

**El §2 del pre-registro dedujo la condición sobre un $q$ GLOBAL, y el modelo elige $q$ POR MUESTRA.**

Con $q$ por muestra, en una pregunta que sí tiene respuesta conviene contestar cuando

$$c \;>\; c^{*} \;=\; \frac{M - F}{1 + M},$$

así que **para que exista un umbral —o sea para que en alguna pregunta convenga callarse— hace falta**

$$\boxed{\;F \;<\; M\;}$$

Con $M = 0{,}5$ y $F = 1{,}5$, el umbral vale $c^{*} = -0{,}667$. **Negativo: nunca conviene callarse.**
El modelo hizo exactamente lo que la pérdida le pedía.

**La compuerta lo tenía delante y no lo leyó como un defecto.** R-6 imprimió *«con F > M, callarse
teniendo la respuesta es peor que errar; si el modelo no logra distinguir, su mejor política es
contestar TODO»*, y eso se registró como **riesgo a vigilar** cuando era una **contradicción de
diseño**: con esos pesos el óptimo no era ambiguo, era el extremo. Un chequeo que calcula $c^{*}$ y
verifica que sea positivo lo habría cerrado antes de gastar GPU.

## 3. Lo que se cambia

**E-1 · $F$ baja de 1,5 a 0,2**, con $L$ y $M$ intactos en 0,5. Ahora $F < M$ y el umbral de confianza
queda en $c^{*} = 0{,}200$: el modelo debería contestar cuando su confianza supera 0,2 y callarse si
no. El óptimo deja de ser un extremo.

**E-2 · Se suma la cross-entropy del valor a la recompensa, con peso 1,0.** No es cosmético y el
cambio de $F$ lo hace necesario:

> En la recompensa, el término que empuja a acertar va multiplicado por $(1-q)$. Un modelo que se
> calla ($q \to 1$) **deja de recibir gradiente hacia la recuperación**, y con $F < M$ callarse vuelve
> a ser lo óptimo cuando $c=0$ —que es correcto, un modelo que no sabe debe callarse—. Sin un término
> que no dependa de $q$, el atractor mudo volvería **y esta vez sin nada que lo saque**.

La CE no depende de $q$ y mantiene vivo el aprendizaje del valor pase lo que pase con la decisión.

**E-3 · La corrida que sigue es EXPLORATORIA y se declara como tal.** Pocas unidades, para mirar los
primeros hitos. **No juzga W-1 ni W-2**, y sus números **no** se usan para elegir pesos otra vez: si
hay que volver a moverlos, eso es otro pre-registro y no un ajuste sobre la marcha.

## 4. Lo que NO se cambia

- **W-2 sigue siendo el criterio principal** y sigue pidiendo exactitud global por encima de 0,4065.
- **El token sigue siendo la condición principal**, por simple y escalable.
- **Los criterios no se aflojan.** Lo que se corrigió es un peso mal derivado, no un umbral que costaba
  alcanzar. La diferencia importa y por eso está escrita.

## 5. Lección, y van cinco

Las cuatro anteriores fueron umbrales fijados sin verificar que fueran alcanzables. **Ésta es de otra
clase y es peor**: un parámetro derivado con la restricción equivocada, con la contradicción impresa
por la propia compuerta y leída como riesgo en vez de como error.

> **Cuando un chequeo dice que el óptimo de la pérdida es un extremo, eso no es un riesgo a vigilar.
> Es un defecto, y hay que cerrarlo antes de correr.**
