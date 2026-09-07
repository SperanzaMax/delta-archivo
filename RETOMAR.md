# Para retomar · lunes 7 de septiembre de 2026

Estado al cierre del 6-sep, 23:00. Bitácora completa en `BITACORA_20260906.md` (copia en
`~/Documentos/Nuevo Transformer/Bitacora/`).

## 1. Lo primero: la campaña quedó a mitad y reanuda sola

Seis unidades entre el paso 500 y 1000 de 2000. Los checkpoints están en `micro_lm/ckpts/` y el
rotador **reanuda con el mismo comando** — no se pierde nada.

```bash
cd micro_lm
PREFIJO=rp SELLO=rel PERT=1 SES_EXTRA=26 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.2 ABST=token \
  MICRO_BATCH=8 BATCH_EVAL=8 SEMBRAR=0 HORIZONTE=6000 \
  ./rotar_abst3.sh 3:0,3:1,3:2 2000 250 250 H K L

PREFIJO=rr SELLO=rel PERT=0 SES_EXTRA=26 KERNEL_Q=5 DONDE=lat2 P_NOSE=0.2 ABST=token \
  MICRO_BATCH=8 BATCH_EVAL=8 SEMBRAR=0 HORIZONTE=6000 \
  ./rotar_abst3.sh 3:0,3:1,3:2 2000 250 250 M N I
```

**No hace falta re-sembrar** (`SEMBRAR=0` y los checkpoints ya existen). Si se re-sembrara se
perdería todo lo entrenado.

Al llegar a 2000, medir con `reloj_o_bandera.py` (R-1) y `geometria_ord.py` (R-3) sobre cada unidad,
y escribir el informe. Los parciales del paso 1000 **ya cumplen las dos** — ver §10 de la bitácora.

## 2. Lo que quedó abierto, en orden

1. **`invento` es lo único que queda de la alucinación.** El SER está en 0,0000-0,0156 con archivo
   largo y `err_identidad` en 0,0000; todo el residuo es contestar cuando no había respuesta. Dos
   ataques conocidos y **sin combinar**: subir `p_nose` y `blanco=error` (que llegó a 1,0000 aislado
   en `b3`).
2. **TinyLlama: el experimento es válido pero el modelo todavía no aprende.** Con el objetivo
   corregido da 0,5000 a 80 pasos. Quedó corriendo 600 pasos; el log es
   `modelo_real/corrida_600.log`. **Ojo: el log sale bufferado, falta `-u` en el comando.**
3. **Fase 2 del sello relativo** (archivo de más de 64 turnos) ya es corrible con
   `datos.lote(turno_base=160)`, condicionada a que R-1 y R-3 cierren en la medición definitiva.
4. **El régimen de 3280 entradas entrenado**, abierto desde el 5-sep.
5. **Confirmar el kernel de la PC**: quedó el 7.0.0-30 por defecto. Si en una o dos semanas no
   vuelve el cuelgue, quedó confirmado que era el -31. La caja negra ahora sobrevive al episodio y
   deja la pila del kernel en `~/.cache/caja-negra/hondo/`.

## 3. Regla que salió del día, y conviene aplicarla antes de medir nada

**Todo instrumento que carga un checkpoint tiene que leer de su config TODO lo que decide la
arquitectura.** Hoy le faltó a dos: `ser.py` no leía `kernel_q` y `reloj_o_bandera.py` no leía
`sello`. Los dos habrían medido con una arquitectura que no era la del checkpoint.

Vale la pena un barrido por el resto de los instrumentos de `micro_lm/` buscando lo mismo.

## 4. Y la otra, que costó cuatro diagnósticos equivocados en un día

Un cero exacto **casi nunca es el modelo**. Hoy fueron: el argmax plano que no puede emitir NOSE, la
tabla del SER que manda los aciertos de las sin-respuesta a otra columna, tres controles de barajado
que no podían fallar, y el objetivo de TinyLlama que era el token de espacio.

**Antes de creerle a un cero: mirar de dónde sale.** Y un bisect donde la variable de interés no es
la única que cambia no bisecta nada.
