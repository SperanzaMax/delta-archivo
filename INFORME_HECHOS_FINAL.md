# Tarea de hechos versionados — resultados

N = 3000 entidades · 10 semillas × 1000 · k = 5 · ε = 0.3 · margen 0.02 · encoder `nomic-embed-text` en minúscula

Prereg + D1 + E1 + enmienda E2, todos congelados con hash antes del dato.

## Métricas por condición (media, IC95 por t de Student, 9 gl)

| condición | VIGENTE | ANTERIOR | COBERTURA |
|---|---|---|---|
| `sin` | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `sobrescritura` | 1.0000 [1.0000, 1.0000] | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] |
| `duplicados` | 0.9992 [0.9987, 0.9997] | 0.9988 [0.9981, 0.9995] | 0.9988 [0.9981, 0.9995] |
| `gemacion` | 0.9928 [0.9910, 0.9946] | 0.9928 [0.9910, 0.9946] | 0.9928 [0.9910, 0.9946] |

## Predicciones pre-registradas

**P1 (principal)** cobertura `gemacion` − `duplicados` = **-0.0060** IC95 [-0.0077, -0.0043] · margen 0.02 → **NO CONFIRMA**

**P2 (control)** ANTERIOR de `sobrescritura` = 0.0000 IC95 [0.0000, 0.0000] → **OK, la tarea mide lo que dice**

**P3** VIGENTE `gemacion` − `sobrescritura` = -0.0072 IC95 [-0.0090, -0.0054] · piso −0.02 → **CUMPLE** (anclar no cuesta precisión sobre el valor al día)

**P4** (ley de escala en K) — no corrida en esta tanda; requiere K ∈ {2,4,8}.

**P5** VersionRAG reporta 58 % en consultas versionadas. Se cita como contexto; el prereg prohíbe declarar superioridad porque la tarea no es idéntica.
