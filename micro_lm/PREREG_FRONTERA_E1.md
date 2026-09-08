# Pre-registro · ESCALON 1 del Micro LM de Frontera · acceso global APRENDIDO

Fecha: 2026-09-08. Autor: Maxi (investigador independiente, ORCID 0009-0005-0413-8554).
Se registra ANTES de correr una sola unidad `attn`.

## 1. Por que existe este experimento

`--donde attn` esta en el codigo desde el 4-sep y **nunca se entreno una unidad con el**. Censo de
los 157 checkpoints al 8-sep: `pre` 63, `lat2` 27, `post` 3, `lat` 3, **`attn` 0**. El control del
4-sep midio acceso global sobre pesos entrenados con `lat2`, que es otra cosa: mide que VE la query,
no que aprende un modelo que la tuvo desde el principio.

La ley de la ventana (`INFORME_QUERY_CIEGA`, kernel 3 -> 5) quedo probada por dos vias: la
correlacional (kernel 5 sube `nose_rel` de 0,61 a 0,99) y la interventiva (con `attn` el cero exacto
desaparece). Falta la tercera y es la que decide el ALCANCE del resultado: **si la causa era el
alcance de la ventana, un modelo que aprende con acceso global tiene que llegar a donde llega el
kernel 5 SIN necesitar un kernel mas grande.** Si no llega, la ventana no era la unica causa.

## 2. Lo que ya esta corrido y hace la comparacion gratis

Dos brazos a 26.000 pasos, mismos hiperparametros, entrenados desde cero:

| unidad | `donde` | kernel_q | alcance de la query | `nose_rel` final |
|---|---|---|---|---|
| `v3_s0/s1/s2`  | lat2 | 3 | 2 tokens, la RELACION cae afuera | **0,6090** (s0) |
| `kq3_s0/s1/s2` | lat2 | 5 | 4 tokens, la relacion entra      | **0,9931** (s0) |
| `at3_s0/s1/s2` | **attn** | — | toda la secuencia causal      | **lo que se pre-registra** |

## 3. La medicion que motiva el brazo nuevo, y que NO es la hipotesis

Hecha hoy antes de lanzar, sobre `kq3_s0`, `rp3_s0` y `v3_s0` (`attn_concentracion.py`,
`attn_concentracion.json`). `attn_causal` usa `q = k = v = x`, y habia que descartar que eso lo
volviera una identidad disfrazada. **No lo es**, y la sospecha queda registrada como refutada:

- la propia posicion se lleva **0,42-0,47** de la masa, no 0,999. La brecha de logits es 3,13 y sale
  de la norma de `ln1(emb[x])` (6,09, no sqrt(D)=11,31: la ganancia aprendida del LayerNorm la baja);
- el resto se reparte entre **9,6 y 11,3 posiciones efectivas** de 24;
- el perfil medio es **PLANO de d1 a d23** (0,0213 a 0,0268, sin decaimiento): acceso global genuino,
  sin sesgo de recencia;
- y **depende del contenido**: TV de cada muestra al perfil medio 0,129 contra un techo de 0,186
  entre muestras distintas, o sea el 70 % de la variabilidad posible.

Lo que queda como limitacion declarada y no como defecto oculto: sin `wq`/`wk` propias el modelo
**no puede aprender a que atender**, solo hereda la geometria de las embeddings, y la brecha de 3,13
hacia la propia posicion no es un peso que el gradiente pueda mover. Por eso este es el escalon 1 y
no el Micro LM de Frontera.

## 4. Hipotesis, en orden de importancia

**H1 (la que decide).** `at3` alcanza `nose_rel` >= 0,95 en las tres semillas, o sea el nivel de
`kq3` y no el de `v3`. Si se cumple, la causa del corte era el ALCANCE y queda probada por las tres
vias. **Se refuta si `nose_rel` final queda por debajo de 0,80 en dos de las tres semillas.**

**H2 (la celda que mas rinde).** La sensibilidad medida con `control_attn.py` sobre los pesos de
`at3` es distinta de cero en las seis distancias Y **su magnitud no decae con la distancia**. Con
pesos de `lat2` el residuo era plano pero debil (TV 0,0097-0,0109 contra 0,0275-0,0444 de la ventana
de kernel 5). La prediccion es que el entrenamiento con acceso global **sube esa magnitud**, y esa
es la firma de que el modelo aprendio a usarlo. Se refuta si la TV de `at3` en `attn` sigue en el
orden de 0,01 y plana, que seria acceso global presente y no aprovechado.

**H3 (barata, sale casi gratis).** `at3` con archivo largo (161 entradas) contra `kq3` con archivo
largo. Si el acceso global tambien baja la interferencia por contenido, los dos cuellos son el mismo;
si no se mueve, son independientes. No condiciona a H1 ni a H2.

## 5. Lo que NO se puede concluir, escrito antes de ver el resultado

`attn` **no contiene a `pre` como caso particular**, a diferencia de `lat2` con `convq` inicializada
en [1,0,...,0]. Un resultado PEOR que `kq3` no se lee como que el acceso global no sirve: se lee como
que este acceso global, sin proyecciones propias y con el sesgo estructural de 3,13 a la propia
posicion, no alcanza. Es el defecto exacto que hundio a `post` el 22-ago y no se va a repetir la
lectura.

## 6. Diseno, congelado

Tres semillas 0/1/2, **desde cero** (`SEMBRAR=0`, igual que `v3` y `kq3`), 26.000 pasos, horizonte de
lr 26.000, d 128, capas 4, lr 1e-3, idioma 2, `p_vieja` 0,35, `p_nose` 0,4, `abst` cabeza,
`sello` abs, `ses_extra` 0, `pert` 0. `KERNEL_Q=5` viaja en la config y **no se usa** en
`donde=attn` (no hay `convq` en el camino); se deja en 5 para que la config quede legible al lado de
`kq3`.

    PREFIJO=at SELLO=abs PERT=0 SES_EXTRA=0 KERNEL_Q=5 DONDE=attn \
      P_NOSE=0.4 ABST=cabeza SEMBRAR=0 HORIZONTE=26000 \
      ./rotar_abst3.sh 3:0,3:1,3:2 26000 2000 500 H K L M N I G C D E F J A

Metrica primaria: `nose_rel` en el paso 26.000. Secundarias: `vigente`, `anterior`, `abstencion`.
Se reporta la corrida entera, no el mejor checkpoint ([[regla-lectura-de-curvas]]).
