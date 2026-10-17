# 01_bass_estimation.py
# steps 3 and 4: GoPro units shipped (look-alike) -> estimate p, q and M
# run this one first, it writes data/bass_parameters.csv and figures 1-3
# the hold-out test and the FAA check are only printed, not saved
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from helper_functions import (bass_cum, bass_adopters_between, fit_bass_cumulative,
                              fit_bass_annual, fit_bass_ols)

ROOT = Path(__file__).parent
DATA = ROOT / "data"
IMG = ROOT / "img"

BLUE, ORANGE, GREY = "#1f4e79", "#d55e00", "#8c8c8c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# data
gp = pd.read_csv(DATA / "gopro_units_shipped.csv")
years = gp["year"].to_numpy()
sales = gp["units_thousands"].to_numpy() / 1000.0      # in million units
cum_all = np.cumsum(sales)

FIRST_YEAR = 2011      # first year where GoPro reports units, this is period 1
MAIN_LAST = 2019       # I stop at 2019 because of covid


def window(last_year):
    # data from 2011 until last_year, t=1 is the end of 2011
    m = years <= last_year
    t = years[m] - FIRST_YEAR + 1
    return t.astype(float), sales[m], cum_all[m]


# main estimation, 2011-2019
t, s, N = window(MAIN_LAST)
fit_cum = fit_bass_cumulative(t, N)
fit_ann = fit_bass_annual(t, s, t_start_first=0.0)
ols = fit_bass_ols(s[1:], N[:-1])

rows = []
for name, f in [("NLS on cumulative units (main)", fit_cum),
                ("NLS on annual units", fit_ann),
                ("Bass (1969) OLS regression", ols)]:
    rows.append({"method": name, "window": f"{FIRST_YEAR}-{MAIN_LAST}",
                 "p": f["p"], "q": f["q"], "M_million": f["M"],
                 "q_over_p": f["q"] / f["p"] if f["p"] else np.nan,
                 "R2": f["R2"],
                 "se_p": f.get("se_p", np.nan), "se_q": f.get("se_q", np.nan),
                 "se_M": f.get("se_M", np.nan),
                 "peak_year": FIRST_YEAR + f["peak_t"] if "peak_t" in f else
                 (FIRST_YEAR + np.log(f["q"] / f["p"]) / (f["p"] + f["q"])
                  if np.isfinite(f["p"]) else np.nan)})
params = pd.DataFrame(rows)


def r2_on_annual(p, q, M):
    # R2 on the yearly units, this one is lower than the R2 on cumulative
    pred = bass_adopters_between(t - 1, t, p, q, M)
    return 1 - np.sum((s - pred) ** 2) / np.sum((s - s.mean()) ** 2)


params["R2_annual_units"] = [r2_on_annual(fit_cum["p"], fit_cum["q"], fit_cum["M"]),
                             r2_on_annual(fit_ann["p"], fit_ann["q"], fit_ann["M"]),
                             r2_on_annual(ols["p"], ols["q"], ols["M"])]
print("\n=== Bass parameters, GoPro look-alike ===")
print(params.round(4).to_string(index=False))
print("OLS coefficients a,b,c:", round(ols["a"], 4), round(ols["b"], 5), round(ols["c"], 6))

# robustness: other end years
rob = []
for last in (2015, 2016, 2019, 2025):
    tt, ss, NN = window(last)
    f = fit_bass_cumulative(tt, NN)
    rob.append({"spec": f"{FIRST_YEAR}-{last}, t=1 in 2011", "n_obs": len(tt), "p": f["p"], "q": f["q"],
                "M_million": f["M"], "R2": f["R2"], "peak_year": FIRST_YEAR + f["peak_t"]})

# robustness: start the clock in july 2009 (first HD gopro) and add the units sold before 2011.
# 0.34 million comes from the S-1 ("more than 8.5 million HD cameras since July 2009"), see report
PRE_2011 = 0.34
for last in (2015, 2019):
    m = years <= last
    tt = (years[m] - 2011 + 2.5).astype(float)     # end of 2011 = 2.5 years after july 2009
    NN = PRE_2011 + np.cumsum(sales[m])
    f = fit_bass_cumulative(tt, NN)
    rob.append({"spec": f"{FIRST_YEAR}-{last}, t=0 in July 2009 (+0.34m pre-2011)", "n_obs": len(tt),
                "p": f["p"], "q": f["q"], "M_million": f["M"], "R2": f["R2"],
                "peak_year": 2009.5 + f["peak_t"]})
rob = pd.DataFrame(rob)
print("\n=== Robustness ===")
print(rob.round(4).to_string(index=False))

# save everything in one csv, first 3 rows are the 3 methods, the rest are the robustness runs
main_rows = params.rename(columns={"method": "spec"})[["spec", "p", "q", "M_million", "R2", "peak_year"]].copy()
main_rows.insert(1, "n_obs", [len(t), len(t), len(t) - 1])
pd.concat([main_rows, rob], ignore_index=True).to_csv(DATA / "bass_parameters.csv", index=False)

# hold-out: fit on 2011-2015 only and see what it predicts for 2016-2019
t5, s5, N5 = window(2015)
f5 = fit_bass_cumulative(t5, N5)
t_h = (years[years > 2015] - FIRST_YEAR + 1).astype(float)
pred_cum = bass_cum(t_h, f5["p"], f5["q"], f5["M"])
pred_ann = np.diff(np.concatenate([[bass_cum(t5[-1], f5["p"], f5["q"], f5["M"])], pred_cum]))
hold = pd.DataFrame({"year": years[years > 2015],
                     "actual_annual_m": sales[years > 2015],
                     "predicted_annual_m": pred_ann,
                     "actual_cum_m": cum_all[years > 2015],
                     "predicted_cum_m": pred_cum})
hold = hold[hold.year <= 2019].copy()
hold["cum_error_pct"] = 100 * (hold.predicted_cum_m / hold.actual_cum_m - 1)
print("\n=== Hold-out (fit 2011-2015, predict 2016-2019) ===")
print(hold.round(3).to_string(index=False))
print("fit on 2011-2015:", {k: round(v, 4) for k, v in f5.items() if k in ("p", "q", "M")})

# figure 1
fig, ax = plt.subplots(figsize=(7.2, 3.8))
cols = [BLUE if y <= MAIN_LAST else GREY for y in years]
ax.bar(years, sales, color=cols, width=0.7)
ax.set_ylabel("Units shipped per year (million)")
ax.set_xticks(years)
ax.set_xticklabels([str(y)[2:] for y in years])
ax.set_xlabel("Year (20xx)")
ax2 = ax.twinx()
ax2.plot(years, cum_all, color=ORANGE, marker="o", ms=3.5, lw=1.8)
ax2.set_ylabel("Cumulative units (million)", color=ORANGE)
ax2.spines["right"].set_visible(True)
ax.set_title("GoPro camera units shipped (SEC filings)\nblue = used for estimation (2011-2019), grey = later years",
             fontsize=10)
fig.tight_layout()
fig.savefig(IMG / "fig1_gopro_units.png", dpi=200)
plt.close(fig)

# figure 2, the fit (left cumulative, right per year)
tg = np.linspace(0, 15, 300)
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.7))
ax = axs[0]
ax.plot(years[years <= MAIN_LAST], cum_all[years <= MAIN_LAST], "o", color=BLUE, label="actual (2011-2019)")
ax.plot(years[years > MAIN_LAST], cum_all[years > MAIN_LAST], "o", color=GREY, mfc="white", label="actual (after 2019, not used)")
ax.plot(tg + FIRST_YEAR - 1, bass_cum(tg, fit_cum["p"], fit_cum["q"], fit_cum["M"]), color=ORANGE, lw=1.8, label="Bass fit")
ax.axhline(fit_cum["M"], color=GREY, ls=":", lw=1)
ax.text(2011.2, fit_cum["M"] + 0.8, f"M = {fit_cum['M']:.1f}m", color=GREY, fontsize=8)
ax.set_xlim(2010, 2026)
ax.set_ylabel("Cumulative units (million)")
ax.set_title("Cumulative adopters", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="lower right")
ax = axs[1]
ax.bar(years[years <= MAIN_LAST], sales[years <= MAIN_LAST], color=BLUE, width=0.7, label="actual (2011-2019)")
ax.bar(years[years > MAIN_LAST], sales[years > MAIN_LAST], color=GREY, width=0.7, label="actual (after 2019)")
tg_a = np.arange(1, 16)
ax.plot(tg_a + FIRST_YEAR - 1,
        bass_adopters_between(tg_a - 1, tg_a, fit_cum["p"], fit_cum["q"], fit_cum["M"]),
        color=ORANGE, marker="o", ms=3.5, lw=1.8, label="Bass fit")
ax.set_xlim(2010, 2026)
ax.set_ylabel("Units per year (million)")
ax.set_title("Adoption per year", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "fig2_bass_fit.png", dpi=200)
plt.close(fig)

# figure 3, FAA drone fleet (used in step 6)
faa = pd.read_csv(DATA / "faa_recreational_drone_fleet.csv")
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.plot(faa["year"], faa["fleet_million_sUAS"], marker="o", color=BLUE, lw=1.8)
ax.set_ylim(0, 2.3)
ax.set_ylabel("Recreational drones in the US (million)")
ax.set_xticks(faa["year"])
ax.set_xticklabels([str(y) for y in faa["year"]], fontsize=8)
for x, y in zip(faa["year"], faa["fleet_million_sUAS"]):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)
ax.set_title("US recreational drone fleet, FAA estimates", fontsize=10)
fig.tight_layout()
fig.savefig(IMG / "fig3_faa_drone_fleet.png", dpi=200)
plt.close(fig)

# extra check on the FAA drone data, this is why I did not use drones as look-alike.
# the series starts in 2016, after the take-off. I fix q to some values and look at the error (SSE),
# if the error is almost the same for different q then p and q can't be separated (see report section 3)
faa_t_years = faa["year"].to_numpy()
faa_N = faa["fleet_million_sUAS"].to_numpy()
print("\n=== FAA drone series: best fit with q fixed (SSE) ===")
for origin in (2012, 2010, 2008):      # I don't know when the drone market really started so I try 3 years
    line = []
    for q_fix in (0.01, 0.06, 0.1, 0.2, 0.3, 0.5):
        best = min((least_squares(lambda th: bass_cum(faa_t_years - origin, th[0], q_fix, th[1]) - faa_N,
                                  [p0, m0], bounds=([1e-7, faa_N.max() * 0.9], [2, 20]))
                    for p0 in (0.001, 0.01, 0.05, 0.1, 0.2) for m0 in (2.0, 2.5, 3.0)),
                   key=lambda r: r.cost)
        line.append(f"q={q_fix}: {2 * best.cost:.4f}")
    print(f"t=0 at end of {origin}:  " + "   ".join(line))

print("\ndone")
