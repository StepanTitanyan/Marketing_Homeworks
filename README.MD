# Homework on Bass Model: diffusion of the Antigravity A1

Marketing course, instructor Karen Hovhannisyan. Student: Stepan Titanyan.

In this homework I look at the **Antigravity A1** (a 360° drone from TIME's *Best Inventions of 2025*, <https://time.com/collections/best-inventions-2025/7318266/antigravity-a1/>) and predict how fast it will spread in the US with the Bass diffusion model. The look-alike innovation is the **GoPro HERO action camera**. The full report with all seven steps and the reference list is in `report/report.pdf` (source: `report/report_source.md`).

## Main results

| | |
|---|---|
| Look-alike data | GoPro camera units shipped 2011 to 2025, from SEC filings |
| Bass parameters (GoPro, 2011 to 2019) | p = 0.031, q = 0.504, M = 41.1 million units |
| Scope for the A1 | United States |
| Market potential of the A1 in the US (Fermi estimate) | 14,000 (low), 51,000 (base), 156,000 (high) buyers |
| Predicted adopters, base case | about 2,200 by end of 2026, 9,800 by end of 2028, peak of about 7,100 a year in 2031, 47,000 by 2035 |

## Files

```
.
├── README.MD
├── requirements.txt         Python packages needed
├── helper_functions.py      Bass formulas and the three estimation methods
├── 01_bass_estimation.py    steps 3-4: estimates p, q, M from the GoPro data (+ robustness, hold-out test)
├── 02_a1_forecast.py        steps 5-7: Fermi estimate of M and A1 adopters by year and quarter
├── data/
│   ├── gopro_units_shipped.csv           GoPro units shipped per year, with the source of each number
│   ├── faa_recreational_drone_fleet.csv  FAA estimate of the US recreational drone fleet, 2016-2025
│   ├── bass_parameters.csv               p, q, M for the three methods and for the robustness runs
│   ├── a1_adopters_by_year.csv           A1 adopters per calendar year (low / base / high M)
│   └── a1_adopters_by_quarter.csv        A1 adopters per calendar quarter, 2025 Q4 to 2028 Q4
├── img/
│   ├── fig1_gopro_units.png              GoPro units per year and cumulative
│   ├── fig2_bass_fit.png                 Bass fit on the GoPro data
│   ├── fig3_faa_drone_fleet.png          US recreational drone fleet (FAA)
│   └── fig4_a1_diffusion_path.png        predicted diffusion path and adopters per year of the A1
└── report/
    ├── report.pdf                        final report
    └── report_source.md                  markdown source of the report
```

## How to run

Python 3.10 or newer.

```
pip install -r requirements.txt
python 01_bass_estimation.py
python 02_a1_forecast.py
```

Run them in this order, because `02_a1_forecast.py` reads `data/bass_parameters.csv`. The hold-out test, the sensitivity table and the side check on the FAA series are printed on screen and are not saved as files.

## Data notes

- **GoPro units** come from the company's S-1 (2014) and 10-K reports (SEC EDGAR); the link for each year is in the csv. Statista was not used because the consumer drone series there starts in 2020 and the numbers are not visible without Premium access. To use another series, replace `data/gopro_units_shipped.csv` (columns `year`, `units_thousands`) and run the scripts again.
- **FAA fleet numbers** are copied from the FAA Aerospace Forecast editions 2017 to 2026 (links in the report and in the csv). They are cumulative registrations times aircraft per registrant, which is the FAA's "high" definition in the 2026 edition.
- **Assumptions.** The Fermi estimate of M has three inputs that are my own assumptions (share of owners who buy, ratio of buyers outside the FAA registry). They are marked in the report and in `02_a1_forecast.py`. I was unable to find any credible data on this.
