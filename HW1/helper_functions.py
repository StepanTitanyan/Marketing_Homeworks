# helper functions for the bass model homework
#
# F(t) = (1 - exp(-(p+q)t)) / (1 + (q/p) * exp(-(p+q)t))
# F is the share of M that already adopted at time t, so adopters so far = M * F(t)
import numpy as np
from scipy.optimize import least_squares


def bass_F(t, p, q):
    t = np.asarray(t, dtype=float)
    e = np.exp(-(p + q) * t)
    return (1 - e) / (1 + (q / p) * e)


def bass_cum(t, p, q, M):
    return M * bass_F(t, p, q)


def bass_adopters_between(t0, t1, p, q, M):
    # new adopters between t0 and t1
    return M * (bass_F(t1, p, q) - bass_F(t0, p, q))


def bass_peak_time(p, q):
    # peak only exists when q > p, otherwise sales just go down from the start
    if q <= p:
        return 0.0
    return np.log(q / p) / (p + q)


def bass_peak_rate(p, q, M):
    if q <= p:
        return M * p
    return M * (p + q) ** 2 / (4 * q)


def _starts(y):
    # many starting values, with only one the optimiser got stuck sometimes
    for p0 in (0.001, 0.005, 0.02, 0.05, 0.1):
        for q0 in (0.2, 0.4, 0.8, 1.5):
            for m0 in (1.2, 2.0, 5.0):
                yield [p0, q0, m0 * y.max()]


def fit_bass_cumulative(t, cum_adopters, q_max=6.0):
    # method 1: nonlinear least squares on the cumulative units
    t = np.asarray(t, float)
    y = np.asarray(cum_adopters, float)

    def resid(th):
        return bass_cum(t, th[0], th[1], th[2]) - y

    lo = [1e-6, 1e-3, y.max()]
    hi = [1.0, q_max, 1000 * y.max()]
    best = None
    for s in _starts(y):
        try:
            r = least_squares(resid, s, bounds=(lo, hi))
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return _pack(best, t, y)


def fit_bass_annual(t_end, annual_sales, t_start_first=0.0, q_max=6.0):
    # method 2: same thing but on the units of each year, not cumulative
    t_end = np.asarray(t_end, float)
    s_obs = np.asarray(annual_sales, float)
    t_prev = np.concatenate([[t_start_first], t_end[:-1]])
    big = s_obs.sum()

    def resid(th):
        return bass_adopters_between(t_prev, t_end, th[0], th[1], th[2]) - s_obs

    lo = [1e-6, 1e-3, big]
    hi = [1.0, q_max, 1000 * big]
    best = None
    for p0 in (0.001, 0.005, 0.02, 0.05, 0.1):
        for q0 in (0.2, 0.4, 0.8, 1.5):
            for m0 in (1.2, 2.0, 5.0):
                try:
                    r = least_squares(resid, [p0, q0, m0 * big], bounds=(lo, hi))
                except Exception:
                    continue
                if best is None or r.cost < best.cost:
                    best = r
    return _pack(best, t_end, s_obs)


def _pack(r, t, y):
    p, q, M = r.x
    n = len(y)
    sse = 2 * r.cost
    sst = float(np.sum((y - y.mean()) ** 2))
    out = {"p": p, "q": q, "M": M, "SSE": sse, "R2": 1 - sse / sst, "n": n,
           "peak_t": bass_peak_time(p, q)}
    # standard errors from the jacobian, not very exact with so few points
    try:
        J = r.jac
        s2 = sse / max(n - 3, 1)
        cov = s2 * np.linalg.inv(J.T @ J)
        se = np.sqrt(np.diag(cov))
    except Exception:
        se = [np.nan] * 3
    out["se_p"], out["se_q"], out["se_M"] = se
    return out


def fit_bass_ols(annual_sales, cum_prev):
    # method 3: the regression from the original Bass paper
    #   S_t = a + b*N_(t-1) + c*N_(t-1)^2
    #   M = (-b - sqrt(b^2 - 4ac)) / (2c),  p = a/M,  q = -M*c
    # if the square root is negative there is no solution and p, q, M stay nan
    S = np.asarray(annual_sales, float)
    N = np.asarray(cum_prev, float)
    X = np.column_stack([np.ones_like(N), N, N ** 2])
    coef, *_ = np.linalg.lstsq(X, S, rcond=None)
    a, b, c = coef
    fitted = X @ coef
    r2 = 1 - np.sum((S - fitted) ** 2) / np.sum((S - S.mean()) ** 2)
    disc = b * b - 4 * a * c
    res = {"a": a, "b": b, "c": c, "R2": r2, "p": np.nan, "q": np.nan, "M": np.nan}
    if disc >= 0 and c != 0:
        roots = [(-b - np.sqrt(disc)) / (2 * c), (-b + np.sqrt(disc)) / (2 * c)]
        roots = [m for m in roots if m > 0]
        if roots:
            M = max(roots)
            res.update({"M": M, "p": a / M, "q": -M * c})
    return res
