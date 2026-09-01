# Metadatos para el envío · preprint del sello de orden

**Title**
Similarity Finds the Fact, Not the Version: A Co-Trained Order Stamp in the Key Resolves Version Conflict in Persistent Model-Internal Memory

**Authors**
Maximiliano Speranza. Independent Researcher, Buenos Aires, Argentina. ORCID 0009-0005-0413-8554.
maximiliano.speranza@gmail.com. Sole author.

**Subject area**
Computer Science / Machine Learning / Natural Language Processing

**Keywords**
language models; persistent memory; retrieval; version conflict; temporal metadata; proactive
interference; pre-registration; small language models

**Funding** None. **Competing interests** None. **Ethics** Not applicable.

**Verificación hecha antes de enviar (1-sep-2026)**
- Los números de las tres tablas verificados contra los JSON crudos de `interno/`, no contra los
  informes. Coinciden exacto, incluidos los por-semilla y las desviaciones.
- Punto de partida (0,9974 / 0,4576) verificado en `resultados_ei2_replica.json`.
- Las cinco semillas de Result 2 (media 0,7667, sd 0,2340) recalculadas a mano desde
  `resultados_ei3c.json` + `resultados_ei3c_extra.json`, con el orden de campos confirmado leyendo
  `ei3c_semillas_extra.py`.
- **Corregido antes de enviar:** la curva de las dos capacidades citaba UNA réplica como si fuera el
  fenómeno. Ahora se dan las dos y se dice qué replica (el orden, no el timing).
- **Corregido antes de enviar:** la bibliografía tenía placeholders. Los nueve identificadores de
  arXiv fueron verificados uno por uno contra arxiv.org, con títulos y autores reales.
- Se quitó un número atribuido a VersionRAG que no se pudo verificar contra la fuente primaria.
- Se agregó la fila del control `shuffled` en Result 3, que estaba en los datos y faltaba en la tabla.
