# ESTADO · 30 de agosto, cierre

Para retomar mañana sin releer nada más que este archivo.

---

## 1. Lo del día, en una frase

> **Las «mudas» no eran ignorantes, y los tres regímenes de la abstención son la misma patología:
> `q` es una CONSTANTE.** Mover el óptimo de la pérdida mueve el valor de esa constante y nada más.

Y de rebote, en el experimento que cerró negativo: **la clave del archivo se comprime 64× sin perder
un punto de recuperación.**

## 2. Campaña `PREREG_RECOMPENSA_L` (SHA `96e750b6`) — CERRADA en la interfaz `token`

Ocho unidades sembradas desde `b3_s3`/`b3_s6` (las declaradas atractor absorbente el 29). Corrieron
las **cuatro de `token`**, que era la condición principal.

| unidad | exactitud | abstención | invento | RECUP |
|---|---:|---:|---:|---:|
| `t03_s3` (L=0) | 0,3020 | 0,4918 | 0,2130 | 0,3675 |
| `t53_s3` (L=0,5) | 0,2938 | 0,4933 | 0,2110 | 0,3364 |
| `t03_s6` (L=0) | 0,3005 | 0,4960 | 0,2105 | 0,3650 |
| `t53_s6` (L=0,5) | 0,3030 | 0,4968 | 0,2102 | 0,3621 |

- **L-1 FALLA 0/4** (piso trivial 0,4065).
- **L-3 CUMPLE 4/4** — primera vez en el proyecto que la abstención cae en un valor intermedio.
- **L-2 da 1 de 2 pares** (+0,0082 y −0,0025), o sea ruido: **confirma la `PRECISION` (`4b61894e`)
  que lo había declarado NO DECIDIBLE antes de mirarlo.**
- **L-6 no se dispara**: RECUP sube respecto del origen (0,3675 contra 0,3654).

**★ Lo que dicen las cuatro juntas:** abstención 0,4918 / 0,4933 / 0,4960 / 0,4968 con `falsa_abst`
~0,48 en todas. **`q` se clava en ~0,5 sin importar semilla, origen ni L.** Llegó al medio y sigue
sin discriminar: se calla en el 48 % de las preguntas que SÍ tienen respuesta.
Ordenados: **mudo 0,4065 > medio-sin-discriminar 0,3020 > locuaz 0,2181.**

**El defecto que explica por qué L-2 no decide** (`PRECISION_RECOMPENSA_L_CE.md`, `4b61894e`): con
`--rec-ce 1.0` la recompensa es el **7,3 %** de la pérdida y el logit de `NOSE` recibe **3,5× menos
gradiente** que un token de valor cualquiera.

> **REGLA que dejó: antes de contrastar dos valores de un peso, medir cuánto gradiente mueve ese peso
> contra el resto de la pérdida. Un contraste sobre el 3 % de la pérdida no es un contraste.**

## 3. `PREREG_CLAVE_DISCRETA` (SHA `3c89348b`) — CIERRA en Fase 0, con un positivo lateral

Idea de Maxi (los embeddings de la memoria «con letras»). CPU sobre `v3_s*`, **cero GPU**.

**Q-0 CUMPLE 3/3 · Q-1 NO CUMPLE 0/3 · Q-2 se dispara en k=256 · Q-3 no interpretable.**
Por el §7 la vía se cierra sin probar una segunda cuantización.

**★★ EL POSITIVO, no buscado: la clave de 128 floats (4096 bits) se reemplaza por 64 BITS de símbolos
con RECUP 1,0000 EXACTO en las tres semillas.** Compresión **64×** sin costo. Es lo único que
sobrevive del experimento y no era lo que iba a medir.

**Lo que hace fuerte al negativo es que Q-0 pasa:** la memoria queda intacta y aun así la ausencia no
tiene firma. **El mejor control: con k=256 el NULO separa MÁS que el tratamiento** (0,726/0,707
contra 0,300/0,272) → ahí el estadístico mide el tamaño del episodio, no coincidencia.

**★ Amplía el cierre del 21-ago:** la objeción viva era que el softmax obliga a leer algo, así que
«ninguna coincidencia» no existía por construcción. Con símbolos el evento **existe**, es observable,
**y sigue sin separar** → la ausencia no vive en la interfaz de memoria por **ninguna** representación.
Sigue siendo calibración.

## 4. El chequeo del ESCALAR (la otra mitad de la idea de Maxi)

«Número con coma: parte entera el tema, decimal la antigüedad».
- **La mitad ya está hecha y ganó**: es el sello de orden de E-I3 (0,4570 → 0,9956), y va en un campo
  **aparte** (`ord[turnos]`, verificado en `modelo.py:230`).
- **Meterlos en el MISMO número no funciona:** con 8 versiones un tema ocupa 0,7 de la recta mientras
  dos temas contiguos están a 0,3 → la distancia DENTRO del tema es **2,3×** la distancia ENTRE temas,
  y con más de 4 versiones las escalas se invierten. Es el defecto que R4 le midió al «eje global».
- **Y el hallazgo que ordena:** las tres codificaciones (escalar, denso+sello, discreta) dan el mismo
  número dígito por dígito en todos los niveles de ruido. **El cuello de botella es inferir el tema, y
  la codificación de la clave no lo toca.**

## 5. Lo primero que hay que hacer mañana

1. **★ LUNES 31 · CORREO INSTITUCIONAL.** Nico Censabella (`nicocensabella@frba.utn.edu.ar`) dijo el
   jueves 27 «espero mañana tener alguna novedad» y no escribió. **Es el trámite de mayor palanca del
   proyecto** — destraba OpenReview (y con él TMLR/ARR/TACL), arXiv y el Scholar buscable.
2. **Decidir dirección**, porque el proyecto **no tiene campaña obligada**:
   - (a) **Cerrar la línea de la pérdida.** Falta la interfaz `cabeza` (`./lanzar_recompensa_L.sh H`,
     4 unidades sembradas y listas en `ckpts/h03_s*.pkl` y `h53_s*.pkl`). El criterio de abandono del
     §6 **pide las dos interfaces**, así que sin esto la línea queda abierta a medias.
   - (b) **Bajar `--rec-ce`** para que la recompensa deje de ser el 7 % de la pérdida. El valor sale
     del **ratio de gradientes medido** (≈3,5), no del desenlace de la corrida de hoy. **Necesita
     pre-registro propio**; elegirlo mirando resultados sería ajustar sobre la marcha.
   - (c) **Los dos papers sin enviar** (el del trípode y el del atractor), que dependen de (1).
3. **Si se retoma la compresión 64×**, es un resultado publicable por sí solo y no tiene campaña
   escrita todavía.

## 6. Estado al cerrar

- **Repo pusheado, `5471b70`.** Working tree limpio, `main` == `origin/main`.
  8 commits hoy: `ad30cb0` · `5206195` · `83518a9` · `766cbce` · `f6c7dc9` · `fdf067b` · `90b68d2` ·
  `5471b70`. `telar-ligamento` también está sincronizado.
- **Nada corriendo**: cero procesos, **cero sesiones de Colab** (verificado en A/J/H/M), sin locks.
  CPU 52 °C.
- **Los `ckpts/` NO están en git** (1,6 GB) pero sí en disco. **Sin ellos se pierde el avance.**
  Los del día: `t03_s3` · `t53_s3` · `t03_s6` · `t53_s6`, más las 4 siembras de `cabeza` sin correr.
- **Bitácora completa en Drive**, 7 archivos, **167.545 bytes verificados byte a byte** contra el
  original: https://drive.google.com/drive/folders/137_f6OtL_NQQj3XDQZ2cghPUGeRF1zoq

## 7. Errores propios de la jornada, los cinco cazados antes de reportar

1. La v1 del chequeo del escalar era un **control vacío** (le pasaba el número de tema exacto a la
   consulta) y dio 1,000 en todo. Mismo defecto que el `m=1` del 12-ago.
2. La v2 imprimió una conclusión que **sus propios números desmentían**.
3. **Q-0 se midió con un PROXY** (coseno) en vez del acierto que el prereg pedía. Al medirlo bien el
   veredicto cambió, y **sin esa corrección se perdía el hallazgo de la compresión 64×**.
4. `tipo` es entero y se comparaba contra strings → Q-3 daba NaN.
5. **Corrección de atribución:** el bug del proceso ZOMBIE que reporté como hallazgo **ya estaba
   diagnosticado el 18-ago**. Lo encontrado en realidad es peor: el arreglo se aplicó a UN script y no
   a sus hermanos, y `tramo_abst.sh` quedó roto once días.
   **Regla: al arreglar un bug de infraestructura, revisar los hermanos en el mismo commit.**
