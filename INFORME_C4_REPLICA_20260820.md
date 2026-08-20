# La réplica en `c4_s0` y `c4_s1` — el presupuesto mejora a las tres, pero R-3 estaba mal escrita

`PREREG_C4_REPLICA.md` (SHA `372d53c8…`, congelado antes de lanzar) · datos en
`c4_s0_presupuesto_20260820.json` y `c4_s1_presupuesto_20260820.json`.

---

## 1 · Los números

Extremos medidos con **2048 muestras** y el generador de prueba (rng 77000+semilla):

| unidad | `falsa_abst` 14000 → 20000 | `nose` | `vigente` | Spearman(`falsa_abst`, paso) |
|---|---|---|---|---:|
| `c4_s0` | 0,0845 → **0,0636** | 0,5893 → 0,7134 | 0,7075 → 0,8053 | −0,2252 (p 0,29) |
| `c4_s1` | 0,0955 → **0,0423** | 0,6532 → 0,7468 | 0,6794 → 0,8374 | −0,2470 (p 0,24) |
| `c4_s2` (19-ago) | 0,1927 → **0,0170** | 0,6704 → 0,6962 | 0,6168 → 0,8793 | −0,3096 (p 0,14) |

- **R-1 CUMPLE en las dos**: `falsa_abst` baja en ambas.
- **R-2 CUMPLE en las dos**: ninguna tendencia creciente; los tres coeficientes son negativos.
- **R-4 CUMPLE en las dos**: `vigente` no cae — sube +0,0978 y +0,1579.
- **R-3 NO EVALUABLE**, ver abajo.

**En las tres unidades de nivel 4, más presupuesto mejora `falsa_abst`, `nose` y `vigente` a la vez.
Ninguna se degrada.** La hipótesis de que la cabeza se rompe al entrenarla de más en tarea difícil
queda descartada en las tres.

## 2 · R-3 estaba mal escrita, y el dato para verlo ya lo teníamos

**El criterio decía:** «alguna de las dos pasa la compuerta a 20000 habiéndola fallado a 14000».

**No puede cumplirse porque su premisa es falsa:** `c4_s0` (0,0845 / 0,5893) y `c4_s1` (0,0955 /
0,6532) **ya pasaban la compuerta a 14000**. No había nada que remontar.

**Y el dato estaba a mano cuando escribí la predicción:** esos dos pares aparecen en la columna
«σ>0,5» de la tabla del `INFORME_SIN_ETIQUETAS_20260820.md`, medidos unas horas antes con las mismas
2048 muestras. Escribí R-3 suponiendo que las tres unidades de nivel 4 fallaban, sin ir a mirar la
tabla que tenía delante. **Es la cuarta vez en el día que un criterio mío pide algo que los datos
disponibles ya desmentían** (las otras: S-4 del corte sin etiquetas, y el §2 del monitor v1).

## 3 · Qué se puede concluir y qué no

**Sí:** el presupuesto mejora las tres unidades de nivel 4, y la degradación que se temía el 19-ago no
existe en ninguna. Eso confirma y extiende el `INFORME_C4S2_20260820.md`.

**No:** que «en nivel 4 la compuerta se falla por presupuesto». **La única unidad de nivel 4 que
fallaba la compuerta era `c4_s2`, y es la única que pasó de fallar a pasar.** Con una sola unidad en
esa condición no hay regla; hay un caso.

**Consecuencia para el confound del §4 del `INFORME_FRONTERA_20260819.md`: queda tocado, no
resuelto.** Sigue siendo cierto que el sub-entrenamiento puede producir fallos que se leerían como
dificultad —`c4_s2` lo muestra— pero esta réplica no puede cuantificar cuánto, porque las otras dos
unidades no estaban fallando. **Para eso haría falta ir a las unidades que sí fallan la compuerta,
que están en las condiciones `token` y `escala`, y ése es el experimento que el §5 del prereg de
`c4_s2` declaró necesario y deliberadamente no autorizó.**

## 4 · Desviación

**D-1.** La primera medición de T-2 sobre `c4_s1` corrió contra un checkpoint del paso **19500**: el
JSON de la corrida ya marcaba 20000 pero el `.pkl` final todavía no había bajado de la VM. Se rehízo
con el de 20000 una vez presente (0,0368 → **0,0423**; el sentido no cambia y la magnitud tampoco
mucho). Queda anotado porque **la espera correcta no es «el JSON dice 20000» sino «el checkpoint dice
20000»**, y es la misma familia de error que la D-1 del corte sin etiquetas: confiar en un indicador
que se actualiza antes que el archivo que uno va a leer.
