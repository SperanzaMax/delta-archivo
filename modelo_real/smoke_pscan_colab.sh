#!/usr/bin/env bash
# ¿Cuanto acelera el scan paralelo EN T4? · 3-sep
#
# En esta PC dio 9,6x (279 -> 29 s/paso) y numericamente equivalente. En GPU el salto deberia ser
# MAYOR, porque alla el camino secuencial no paga computo sino 192*24 = 4608 lanzamientos de kernel
# por forward, que es justo lo que el scan paralelo borra. Este numero decide todo lo demas:
# cuantos pasos entran en una sesion, cuantos hechos, y si vuelve a estar en juego el 370m.
#
#   Uso:  smoke_pscan_colab.sh <CUENTA>
set -uo pipefail
AQUI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$AQUI"
CUENTA="${1:?falta la cuenta}"
COLAB=/home/maxi/.venv-colab-cli/bin/colab
POOL="$HOME/.colab-pool"

if [ "$CUENTA" = "A" ]; then CL=( "$COLAB" --auth adc ); unset CLOUDSDK_CONFIG
else export CLOUDSDK_CONFIG="$HOME/.gcloud-cuenta$CUENTA"; CL=( "$COLAB" --auth adc --config "$HOME/.colab-cuenta$CUENTA.json" ); fi

lk="$POOL/en_uso_$CUENTA"
if [ -f "$lk" ] && kill -0 "$(cat "$lk" 2>/dev/null)" 2>/dev/null; then
  echo "cuenta $CUENTA ocupada por el pid $(cat "$lk")"; exit 1
fi
echo $$ > "$lk"; trap 'rm -f "$lk"' EXIT

VIVAS="$(timeout -k 20 180 "${CL[@]}" sessions 2>/dev/null | grep -iE "T4|L4|GPU" || true)"
SESION="$(echo "$VIVAS" | head -1 | sed -n 's/^\[\([^]]*\)\].*/\1/p')"
if [ -n "$SESION" ]; then
  echo "== smoke en $CUENTA · REUSA $SESION"
else
  SESION="smk_${CUENTA,,}_$(date +%H%M)"
  echo "== smoke en $CUENTA · sesion NUEVA $SESION"
  OUT="$(timeout -k 30 600 "${CL[@]}" new -s "$SESION" --gpu T4 2>&1 | tail -3)"
  echo "$OUT"
  echo "$OUT" | grep -qi "READY" || { echo "sin sesion en $CUENTA"; exit 1; }
fi

TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"; rm -f "$lk"' EXIT
tar czf "$TMP/real.tgz" tarea_real.py entrenar_real.py vocabulario.json
timeout -k 30 300 "${CL[@]}" upload -s "$SESION" "$TMP/real.tgz" /content/real.tgz || exit 1

cat > "$TMP/medir.py" <<'PYOUT'
import os, subprocess, sys, textwrap
os.makedirs('/content/real', exist_ok=True)
subprocess.run('tar xzf /content/real.tgz -C /content/real', shell=True, check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'mambapy'], check=False)

guion = textwrap.dedent('''
    import os, sys, time
    import numpy as np, torch
    sys.path.insert(0, "/content/real")
    import tarea_real as T
    from transformers import AutoTokenizer, AutoModelForCausalLM
    try:
        from mambapy.pscan import pscan; print("mambapy IMPORTA OK", flush=True)
    except Exception as e:
        print("mambapy NO importa:", e, flush=True)
    print("gpu", torch.cuda.get_device_name(0), flush=True)

    MOD = os.environ.get("MOD", "state-spaces/mamba-130m-hf")
    tok = AutoTokenizer.from_pretrained(MOD)

    def cargar(pscan_on, ckpt):
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_pretrained(MOD, dtype=torch.float32)
        m.config.use_mambapy = bool(pscan_on)
        for c in m.backbone.layers:
            c.mixer.use_mambapy = bool(pscan_on)
        m.cuda().train()
        if ckpt:
            m.gradient_checkpointing_enable(); m.config.use_cache = False
        return m

    # --- equivalencia en GPU, mismo control que en la PC
    rng = np.random.default_rng(7)
    ids, lab, _, _ = T.lote(rng, tok, 2, ("directa",), n_hechos=4, largo=48)
    ids, lab = ids.cuda(), lab.cuda()
    sal = {}
    for nom, fl in (("sec", False), ("psc", True)):
        m = cargar(fl, False)
        o = m(ids, labels=lab); o.loss.backward()
        sal[nom] = (o.logits.detach().clone(), float(o.loss.detach()),
                    m.backbone.layers[0].mixer.x_proj.weight.grad.detach().clone())
        del m; torch.cuda.empty_cache()
    dlg = float((sal["sec"][0]-sal["psc"][0]).abs().max()); elg = float(sal["sec"][0].abs().max())
    dg  = float((sal["sec"][2]-sal["psc"][2]).abs().max()); eg  = float(sal["sec"][2].abs().max())
    print(f"EQUIV logits rel {dlg/elg:.3e} · loss {sal['sec'][1]:.6f} vs {sal['psc'][1]:.6f} "
          f"· grad rel {dg/max(eg,1e-12):.3e}", flush=True)

    # --- velocidad, las dos ramas, con el mismo montaje de la campania real
    B, LARGO, NH = 8, 192, 16
    for nom, fl in (("SECUENCIAL", False), ("PSCAN", True)):
        try:
            m = cargar(fl, True)
            opt = torch.optim.AdamW(m.parameters(), lr=3e-5)
            rng = np.random.default_rng(0)
            n = 2 if not fl else 3
            for _ in range(2):
                i2, l2, _, _ = T.lote(rng, tok, B, ("directa",), n_hechos=NH, largo=LARGO)
                m(i2.cuda(), labels=l2.cuda()).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(n):
                i2, l2, _, _ = T.lote(rng, tok, B, ("directa",), n_hechos=NH, largo=LARGO)
                m(i2.cuda(), labels=l2.cuda()).loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); dt = (time.time()-t0)/n
            pk = torch.cuda.max_memory_allocated()/2**30
            print(f"VEL {nom}: {dt:.3f} s/paso · pico {pk:.2f} GiB · "
                  f"1200 pasos = {dt*1200/3600:.2f} h", flush=True)
            del m, opt; torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            print(f"VEL {nom}: FALLO {type(e).__name__}: {str(e)[:200]}", flush=True)
''')
open('/content/medir_guion.py', 'w').write(guion)
log = open('/content/smoke.log', 'w')
p = subprocess.Popen([sys.executable, '-u', '/content/medir_guion.py'],
                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
open('/content/smoke.pid', 'w').write(str(p.pid))
print('lanzado pid', p.pid, flush=True)
PYOUT
timeout -k 30 900 "${CL[@]}" exec -s "$SESION" --timeout 600 -f "$TMP/medir.py" || exit 1

cat > "$TMP/ver.py" <<'PY'
import os
try:
    pid = int(open('/content/smoke.pid').read())
    vivo = os.path.exists('/proc/%d' % pid)
    if vivo:
        try:
            if open('/proc/%d/stat' % pid).read().split(')')[-1].split()[0] == 'Z': vivo = False
        except Exception: pass
    print('VIVO=', vivo)
except Exception as e:
    print('VIVO= ?', e)
try:
    print('LOG:\n' + ''.join(open('/content/smoke.log').readlines()[-14:]))
except Exception:
    print('(sin log)')
PY

for i in $(seq 1 30); do          # hasta 30 min
  sleep 60
  OUT="$(timeout -k 20 180 "${CL[@]}" exec -s "$SESION" --timeout 120 -f "$TMP/ver.py" 2>&1 | tail -18)"
  echo "$OUT"
  echo "$OUT" | grep -q "VIVO= False" && break
done
timeout -k 30 300 "${CL[@]}" download -s "$SESION" "/content/smoke.log" "$AQUI/smoke_pscan_${CUENTA}.log" 2>&1 | tail -1
echo "== SESION $SESION en la cuenta $CUENTA queda VIVA a proposito, para reusarla en la campania"
