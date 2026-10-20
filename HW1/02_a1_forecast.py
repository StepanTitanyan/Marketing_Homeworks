# 02_a1_forecast.py
# steps 5, 6, 7: A1 diffusion path in the US, market potential M (fermi) and adopters per period
# run 01_bass_estimation.py before this, it needs data/bass_parameters.csv
# writes data/a1_adopters_by_year.csv, data/a1_adopters_by_quarter.csv and img/fig4
# the sensitivity table is only printed
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from helper_functions import bass_F, bass_peak_time, bass_peak_rate

ROOT = Path(__file__).parent
DATA = ROOT / "data"
IMG = ROOT / "img"

BLUE, ORANGE, GREY, GREEN = "#1f4e79", "#d55e00", "#8c8c8c", "#2e7d32"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

LAUNCH = date(2025, 12, 4)      # A1 went on sale
TODAY = date(2026, 10, 18)
PRICE_USD = 1599                # cheapest bundle


def yrs_since_launch(d):
    return (d - LAUNCH).days / 365.25


# p and q from the look-alike (read from the csv of script 01)
par = pd.read_csv(DATA / "bass_parameters.csv").set_index("spec")
p_main, q_main = par.loc["NLS on cumulative units (main)", ["p", "q"]]
p_ols, q_ols = par.loc["Bass (1969) OLS regression", ["p", "q"]]
p_early, q_early = par.loc["2011-2015, t=1 in 2011", ["p", "q"]]
p_meta, q_meta = 0.03, 0.38      # average values from Sultan, Farley and Lehmann (1990)

pq_sets = {
    "Main: GoPro 2011-2019 (NLS)": (p_main, q_main),
    "GoPro 2011-2015 only": (p_early, q_early),
    "GoPro 2011-2019 (Bass OLS)": (p_ols, q_ols),
    "Meta-analysis average (Sultan et al. 1990)": (p_meta, q_meta),
}

# fermi estimate of M for the US
# the two FAA numbers are real data (FAA 2026): owners with 1-2 aircraft and owners with 3+ aircraft
n_casual = 296_277
n_enthusiast = 26_802
scen = ["low", "base", "high"]
buy_enth = dict(low=0.15, base=0.30, high=0.50)      # assumption, my guess
buy_casual = dict(low=0.01, base=0.03, high=0.06)    # assumption, my guess
ratio_new = dict(low=1.0, base=2.0, high=4.0)        # assumption: buyers outside the FAA registry per registry buyer

M = {}
fermi_rows = []
for s in scen:
    pool_a = n_enthusiast * buy_enth[s] + n_casual * buy_casual[s]
    M[s] = pool_a * (1 + ratio_new[s])
    fermi_rows.append((s, pool_a, M[s]))
print("\nFermi: pool A (existing drone owners who buy) and M (US) per scenario")
for s, a, m in fermi_rows:
    print(f"  {s:5s} pool A = {a:9.0f}   M = {m:9.0f}   lifetime US revenue = ${m * PRICE_USD / 1e6:6.1f}m")

t_launch_grid = np.linspace(0, 15, 600)

# adopters per calendar year, the first period is only 4 Dec - 31 Dec 2025
bounds = [LAUNCH] + [date(y, 12, 31) for y in range(2025, 2036)]
labels = ["2025 (from 4 Dec)"] + [str(y) for y in range(2026, 2036)]
t_b = np.array([yrs_since_launch(d) for d in bounds])
rows = []
for i, lab in enumerate(labels):
    row = {"period": lab, "t_start_years": t_b[i], "t_end_years": t_b[i + 1] if i + 1 < len(t_b) else np.nan}
    t0, t1 = t_b[i], t_b[i + 1]
    row["cum_share_end"] = bass_F(t1, p_main, q_main)
    for s in scen:
        row[f"adopters_{s}"] = M[s] * (bass_F(t1, p_main, q_main) - bass_F(t0, p_main, q_main))
        row[f"cumulative_{s}"] = M[s] * bass_F(t1, p_main, q_main)
    rows.append(row)
yearly = pd.DataFrame(rows)
yearly.to_csv(DATA / "a1_adopters_by_year.csv", index=False)
print("\n=== A1 adopters per calendar year (US) ===")
print(yearly[["period", "cum_share_end", "adopters_low", "adopters_base", "adopters_high",
              "cumulative_base"]].round(3).to_string(index=False))

# same by quarter, 2025 Q4 until 2028 Q4
q_ends = [date(2025, 12, 31)]
for y in (2026, 2027, 2028):
    for md in ((3, 31), (6, 30), (9, 30), (12, 31)):
        q_ends.append(date(y, *md))
q_bounds = [LAUNCH] + q_ends
q_labels = ["2025 Q4 (from 4 Dec)"] + [f"{y} Q{k}" for y in (2026, 2027, 2028) for k in (1, 2, 3, 4)]
tq = np.array([yrs_since_launch(d) for d in q_bounds])
qrows = []
for i, lab in enumerate(q_labels):
    t0, t1 = tq[i], tq[i + 1]
    r = {"period": lab, "t_end_years": t1}
    for s in scen:
        r[f"adopters_{s}"] = M[s] * (bass_F(t1, p_main, q_main) - bass_F(t0, p_main, q_main))
        r[f"cumulative_{s}"] = M[s] * bass_F(t1, p_main, q_main)
    qrows.append(r)
quarterly = pd.DataFrame(qrows)
quarterly.to_csv(DATA / "a1_adopters_by_quarter.csv", index=False)
print("\n=== A1 adopters per calendar quarter (US) ===")
print(quarterly[["period", "adopters_low", "adopters_base", "adopters_high", "cumulative_base"]].round(0).to_string(index=False))

# where are we today
t_today = yrs_since_launch(TODAY)
print(f"\nToday = {TODAY}, t = {t_today:.3f} years since launch, share of M adopted = {bass_F(t_today, p_main, q_main):.4f}")
for s in scen:
    print(f"   cumulative adopters to date, {s}: {M[s] * bass_F(t_today, p_main, q_main):,.0f}")

# peak of the adoption
tp = bass_peak_time(p_main, q_main)
print(f"Peak of adoption rate at {tp:.2f} years after launch "
      f"({date.fromordinal(LAUNCH.toordinal() + int(tp * 365.25))}), "
      f"peak rate = {bass_peak_rate(p_main, q_main, 1):.3f} x M per year "
      f"-> base {bass_peak_rate(p_main, q_main, M['base']):,.0f} per year")
# when are 50% and 90% of M reached
for share in (0.5, 0.9):
    tt = t_launch_grid[np.argmax(bass_F(t_launch_grid, p_main, q_main) >= share)]
    print(f"{share:.0%} of M reached after {tt:.1f} years")

# what happens with other p and q
sens_rows = []
for name, (pp, qq) in pq_sets.items():
    r = {"parameters": name, "p": pp, "q": qq, "q_over_p": qq / pp, "peak_after_years": bass_peak_time(pp, qq)}
    for yy in (1, 3, 5, 10):
        r[f"share_of_M_after_{yy}y"] = bass_F(yy, pp, qq)
        r[f"base_M_cum_units_after_{yy}y"] = M["base"] * bass_F(yy, pp, qq)
    sens_rows.append(r)
sens = pd.DataFrame(sens_rows)
print("\n=== sensitivity to (p,q) ===")
print(sens.round(3).to_string(index=False))

# figure 4, left = F(t) for the 4 sets of p and q, right = adopters per year for low/base/high M
fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.8))
ax = axs[0]
styles = [(BLUE, "-", 2.2), (ORANGE, "--", 1.4), (GREEN, "-.", 1.4), (GREY, ":", 1.8)]
for (name, (pp, qq)), (c, ls, lw) in zip(pq_sets.items(), styles):
    ax.plot(t_launch_grid, 100 * bass_F(t_launch_grid, pp, qq), color=c, ls=ls, lw=lw,
            label=f"{name}\n(p={pp:.3f}, q={qq:.2f})")
ax.set_xlim(0, 15)
ax.set_xlabel("Years since launch (Dec 2025)")
ax.set_ylabel("Share of market potential adopted (%)")
ax.set_title("Diffusion path F(t)", fontsize=10)
ax.legend(frameon=False, fontsize=6.5, loc="lower right")
ax = axs[1]
tt = np.arange(1, 16)
ann = M["base"] * (bass_F(tt, p_main, q_main) - bass_F(tt - 1, p_main, q_main))
ann_lo = M["low"] * (bass_F(tt, p_main, q_main) - bass_F(tt - 1, p_main, q_main))
ann_hi = M["high"] * (bass_F(tt, p_main, q_main) - bass_F(tt - 1, p_main, q_main))
ax.bar(tt, ann / 1000, color=BLUE, width=0.7, label="base (M = %.0fk)" % (M["base"] / 1000))
ax.plot(tt, ann_lo / 1000, color=GREY, marker="o", ms=3, lw=1, label="low (M = %.0fk)" % (M["low"] / 1000))
ax.plot(tt, ann_hi / 1000, color=ORANGE, marker="o", ms=3, lw=1, label="high (M = %.0fk)" % (M["high"] / 1000))
ax.set_xlabel("Year since launch (year 1 = Dec 2025 - Nov 2026)")
ax.set_ylabel("New A1 adopters in the US per year (thousand)")
ax.set_title("Adopters per year", fontsize=10)
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(IMG / "fig4_a1_diffusion_path.png", dpi=200)
plt.close(fig)

print("\ndone")
