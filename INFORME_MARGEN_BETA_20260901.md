# La señal de «no está» NO vive en la búsqueda · el afilado no tiene margen

**2026-09-01.** Pregunta de Maxi: *«tiene que haber algo en el aprendizaje de BÚSQUEDA que podamos
modificar o intervenir para que le sea más fácil decir la verdad»*. `margen_beta.py`, dos unidades,
n=1024, CPU, sin GPU. **Compuerta previa a cualquier entrenamiento**, y cierra la vía barata.

## 1. La idea que se probó

`NOTA_BUSQUEDA_UNIFORME_20260831.md` dejó medido que la búsqueda «busca siempre igual y a media
máquina», porque el divisor de `sim = q·k/√d` es una **constante**. La intervención natural es un
**`β(x)` aprendido por consulta**: afilar cuando encuentra y difuminar cuando no, para que «no está»
tenga firma en la lectura misma.

Antes de entrenar eso hay que saber si tiene **margen**: ¿existe **algún** `β` fijo con el que la
forma de la lectura separe «está» de «no está»? Si ninguno separa, un `β(x)` aprendido tampoco puede.

## 2. Ninguno separa, y replica

| `β` | AUC entropía (`p3_s0`) | AUC entropía (`n3_s0`) | brecha en σ |
|---:|---:|---:|---:|
| 0,25 | 0,4734 | 0,4784 | −0,081 |
| **1,00 (el de hoy)** | **0,4741** | **0,4778** | −0,081 |
| 4,00 | 0,4761 | 0,4758 | −0,070 / −0,090 |
| 8,00 | 0,4956 | 0,4774 | −0,015 / −0,085 |
| 16,00 | 0,4868 | 0,4792 | −0,031 / −0,065 |

**Todos en el azar o por debajo**, en las dos unidades, con `β` barrido en un factor de 64. Y la
brecha es **negativa**: la lectura es *levemente menos* dispersa cuando la respuesta **no** está, al
revés de lo que la idea predice, y de tamaño despreciable (−0,08 σ).

**Dato que agrega la medición:** con 6 entradas por muestra la entropía máxima posible es **1,7916** y
la búsqueda vive en **1,7697**. No es que busque «a media máquina»: **está al 98,8 % de lo uniforme**.
Reparte la masa casi por igual entre todas las entradas, siempre.

> **El afilado de la búsqueda no tiene margen. Un `β(x)` aprendido no puede extraer una señal que no
> está en la distribución que modifica.** La vía queda cerrada por 12 minutos de CPU, sin GPU.

## 3. ★ Lo que el negativo enseña, y vale más que la idea que descarta

`p3_s0` **sabe abstenerse** —`nose` 0,9674 con `falsa_abst` 0,0069— **y su búsqueda es indistinguible
entre presente y ausente.** Las dos cosas a la vez.

> **Entonces el modelo que ya resuelve la tarea NO usa la forma de la búsqueda para saber que algo no
> está.** Lo detecta **aguas abajo**, en el estado, después de recuperar.

Eso ordena tres números que hasta hoy estaban sueltos, y la escalera es monótona:

| dónde se mira | AUC contra la ausencia |
|---|---:|
| la **búsqueda** (forma de la lectura, cualquier `β`) | **≈ 0,48** — azar |
| el **estado**, capa por capa | 0,50 → 0,667 → 0,683 → 0,685 |
| el **estado final** (techo medido el 31-ago con 5 lectores) | **0,7003** |

**La señal de ausencia no existe en la lectura y se construye recién después.** Por eso ninguna
intervención sobre *cómo busca* la va a crear: lo que hay que mejorar es **qué hace con lo que
recuperó**, que es donde la señal aparece.

## 4. Lo que no dice

- Dos unidades, un nivel, `β` **global** por corrida. Un `β(x)` que dependa de la consulta podría, en
  principio, hacer algo que ningún `β` fijo hace — pero tendría que **crear** la señal, no
  amplificarla, y para eso ya no es un afilado.
- Sólo mira **entropía** y **masa de la ganadora**. Otra funcional de la lectura podría separar; la
  que la idea proponía es ésta.
- No toca la **query** (`qr`), sólo la temperatura. Buscar en otra **dirección** sigue abierto, y es
  la versión fuerte de la idea del desacuerdo (entidad contra relación).
