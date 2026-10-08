"""Levenberg-Marquardt in numpy. This box has no scipy and installing one system-wide
is not ours to do; the fits here have ten parameters and a few thousand residuals."""

import numpy as np


def lm(fun, x0, iters=80, lam=1e-2, rel_step=1e-4, verbose=False, tol=1e-9, max_step=1.0):
    """Minimise sum(fun(x)**2). Forward-difference Jacobian, multiplicative damping.
    Parameters are expected in log space, so a step is clipped to `max_step` per
    coordinate: one bad Jacobian must not throw a capacity to 1e200.

    The difference step is `rel_step` times max(|x|, 1), never less than `rel_step`
    itself: a parameter that starts at 1 has log 0, and a step scaled by |x| alone
    would be too small to see it move."""
    x = np.array(x0, float)
    r = fun(x)
    cost = float(r @ r)
    for it in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            h = rel_step * max(abs(x[j]), 1.0)
            xj = x.copy()
            xj[j] += h
            J[:, j] = (fun(xj) - r) / h
        A, g = J.T @ J, J.T @ r
        moved = False
        for _ in range(14):
            try:
                step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            step = np.clip(step, -max_step, max_step)
            r2 = fun(x + step)
            c2 = float(r2 @ r2)
            if np.isfinite(c2) and c2 < cost:
                gain = cost - c2
                x, r, cost, lam, moved = x + step, r2, c2, lam / 3, True
                break
            lam *= 4
        if verbose:
            print(f"    lm {it:3d}: rms {np.sqrt(cost / len(r)):.5f}  lam {lam:.1e}", flush=True)
        if not moved or gain < tol * max(cost, 1e-30):
            break
    return x, cost
