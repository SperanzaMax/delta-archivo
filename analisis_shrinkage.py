"""Analisis del espacio con estimador robusto de covarianza (Ledoit-Wolf).

Con d=2048 y n=10.000 la razon n/d es ~4.9: la covarianza muestral ya no es singular, pero sus
autovalores siguen sesgados (regimen Marchenko-Pastur). El shrinkage de Ledoit-Wolf corrige ese
sesgo encogiendo la muestral hacia un objetivo isotropico, con un coeficiente estimado de los
propios datos.

Se reportan las DOS estimaciones. Si la dimension efectiva cambia poco entre ambas, la conclusion
no depende del estimador y es defendible; si cambia mucho, hay que reportar la corregida.

Uso: python analisis_shrinkage.py embeddings_10k.npy
"""
import sys
import numpy as np

RUTA = sys.argv[1] if len(sys.argv) > 1 else "embeddings_10k.npy"


def dim_efectiva(lam):
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-30))


def ledoit_wolf(X):
    """Shrinkage analitico hacia mu*I. Devuelve (autovalores, coeficiente de shrinkage)."""
    n, d = X.shape
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / n
    mu = np.trace(S) / d
    T = mu * np.eye(d)
    d2 = np.sum((S - T) ** 2) / d
    b2 = sum(np.sum((np.outer(x, x) - S) ** 2) for x in Xc[:min(n, 2000)])
    b2 = b2 / (min(n, 2000) ** 2 * d)
    b2 = min(b2, d2)
    rho = b2 / d2 if d2 > 0 else 0.0
    Sh = rho * T + (1 - rho) * S
    return np.linalg.eigvalsh(Sh), float(rho)


def main():
    X = np.load(RUTA).astype(np.float64)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    n, d = X.shape
    print(f"{RUTA}: n={n}, d={d}, n/d={n/d:.2f}\n")

    rng = np.random.default_rng(0)
    i, j = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    k = i != j
    cos = np.sum(X[i[k]] * X[j[k]], 1)
    print(f"|cos| medio entre pares : {np.mean(np.abs(cos)):.4f}  (sd {np.std(cos):.4f})")
    print(f"norma del vector medio  : {np.linalg.norm(X.mean(0)):.4f}")

    Xc = X - X.mean(0)
    lam_emp = np.linalg.eigvalsh(np.cov(Xc.T))
    lam_lw, rho = ledoit_wolf(X)

    print(f"\n{'estimador':>28} {'dim efectiva':>14}")
    print(f"{'covarianza muestral':>28} {dim_efectiva(lam_emp):14.1f}")
    print(f"{'Ledoit-Wolf (rho=%.3f)' % rho:>28} {dim_efectiva(lam_lw):14.1f}")

    ref = rng.normal(size=(n, d))
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    lam_ref = np.linalg.eigvalsh(np.cov((ref - ref.mean(0)).T))
    print(f"{'uniforme (referencia)':>28} {dim_efectiva(lam_ref):14.1f}")

    tot = np.clip(lam_emp, 0, None).sum()
    acum = np.cumsum(np.sort(np.clip(lam_emp, 0, None))[::-1]) / tot
    for frac in (0.5, 0.9, 0.99):
        print(f"componentes para {frac:.0%} de la varianza: "
              f"{int(np.searchsorted(acum, frac)) + 1}")


if __name__ == "__main__":
    main()
