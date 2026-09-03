# Envío a TMLR, paso a paso · todo resuelto salvo apretar el botón

**Entrar a** `openreview.net`, buscar **TMLR** en Active Venues y darle a **Submit**.

Antes de empezar, tener a mano estos dos archivos, los dos en `preprint/ventana/`:

| campo del formulario | archivo |
|---|---|
| PDF del manuscrito | `tmlr/ventana_tmlr.pdf` (9 páginas, anónimo) |
| Supplementary material | `suplementario_anonimo.zip` (102 KB, 39 archivos) |

⚠️ **El PDF que va es el de `tmlr/`, NO `ventana_en.pdf`.** El de la carpeta de arriba lleva tu
nombre, tu ORCID, tu mail y el link al repo con tu usuario de GitHub. Subir ése rompe el doble ciego
en el primer renglón.

---

## 1 · Title

```
The Query Cannot See the Question: A Short Convolution's Reach Decides Which Part of a Query Conditions Retrieval
```

## 2 · Authors

Los toma solo de tu perfil de OpenReview. **No escribir nada** y no agregar coautores.

## 3 · Abstract

En texto plano, sin LaTeX. OpenReview acepta `$...$` para matemática simple pero no `\textbf`,
`\emph` ni `\texttt`. Está listo para copiar en `METADATOS_ARXIV.md`, sección Abstract, que es el
mismo texto.

## 4 · Keywords

```
state space models, linear attention, associative recall, abstention, selective prediction, mechanistic interpretability, retrieval, pre-registration
```

## 5 · Supplementary material

Subir `suplementario_anonimo.zip`. Auditado: cero coincidencias de nombre, ORCID, mail, dominio
institucional o rutas personales. TMLR exige que el suplementario esté anonimizado igual que el
manuscrito, y lo está.

**No** subir el link a `github.com/SperanzaMax/delta-archivo`, que te identifica.

## 6 · Lo que NO hay que declarar

**No pegar el enlace de arXiv** en ningún campo, ni ahora ni cuando exista. El doble ciego se
mantiene, textual, *«by not linking to another version that includes the authors' names»*. Que el
paper esté en arXiv está permitido; linkearlo desde el envío, no.

## 7 · Conflictos de interés

Los toma del perfil de OpenReview, donde ya cargaste la UTN. Nada que hacer.

## 8 · Action editor

Si pide sugerencias, elegir de la lista viva a alguien cuyo perfil diga **modelos de espacio de
estados / atención lineal** o **interpretabilidad mecanicista**. No elegir por prestigio: el criterio
de TMLR es si los claims están sostenidos, y eso lo juzga mejor quien conoce la arquitectura.

---

## Después de enviar

| cuándo | qué pasa |
|---|---|
| 1 semana | te asignan action editor. Es lo único que TMLR compromete por escrito |
| ~2 semanas más | llegan las tres reviews, públicas en OpenReview |
| hasta 5 semanas | discusión con los revisores. **Acá sí hay que responder**, no es pasivo |
| **~91 días** | decisión. Es la mediana real de 2025 para papers cortos; el objetivo declarado es 9 semanas |

**El paper tiene 9 páginas, o sea que entra como corto.** Los de más de 12 van a 104 días y sus
revisores tienen 4 semanas en vez de 2.

## Y lo que hay que tener presente al apretar el botón

La cuota es la *Generalized Harmonic Quota Rule* con N₁ = 2: firmando solo, **dos envíos al año**. Y
textual, *«Budget is spent for all submissions by an author, including those which are desk
rejected»*. Este es el primero de los dos.
