"""Workplace parking pricing and commuting mode: reproducible analysis script.

Inputs : parking_data.csv, transit_access.csv (same folder as this script).
Outputs: output/tables/*.csv and output/tables/*.md (tables A1, B1-B3, C1-C3, D1, E1; table_A1_communes = full list
         of the 39 municipalities, Table A1 in the report being a summary by site and class),
         output/figures/figure_D1.pdf and .png, output/results.txt (numbers quoted in the text).

Run: python analysis.py

Rounding conventions (identical throughout the report)
- shares and rates in %        : 1 decimal
- changes in percentage points : 1 decimal
- elasticities                 : 2 decimals
- regression coefficients      : 3 decimals (standard errors in parentheses)
- days per week                : 2 decimals
- CHF amounts                  : rounded to the franc (revenue to the thousand CHF)
- burden (% of income)         : 2 decimals
- decimal separator            : point everywhere (tables and text); thousands separated by "'" (80'800 CHF)
- transit index (Table A1)     : score out of 100 with 1 decimal, PT/car ratio with 2 decimals, times in minutes
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output"
TAB = OUT / "tables"
FIG = OUT / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

WEEKS = 46                      # working weeks per year
FEE_2024, FEE_2025 = 2.0, 4.0   # Lausanne fee
SCENARIOS = [4, 5, 6, 8, 10]    # CHF per day, 4 = 2025 baseline
RECO_FEE = 6                    # recommended 2026 fee (Part F)
N_BOOT = 999
RNG = np.random.default_rng(2026)

LOG = []


def note(*args):
    """Record a result quoted in the text."""
    line = " ".join(str(a) for a in args)
    LOG.append(line)
    print(line)


def save(df, name, index=False):
    df.to_csv(TAB / f"{name}.csv", index=index)
    d = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * d.shape[1]]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n")


def chf(x):
    """CHF amount rounded to the franc, thousands separated by "'" (the point is reserved for decimals)."""
    return f"{x:,.0f}".replace(",", "'")


def pct(x):
    return f"{100 * x:.1f}"


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# =============================================================================
# 0. Loading and cleaning
# =============================================================================
raw = pd.read_csv(ROOT / "parking_data.csv")
df = raw.copy()

# 0.1 Municipalities: 62 labels -> 39 municipalities (inconsistent case and hyphens)
def commune_key(s):
    return s.str.upper().str.replace("-", " ", regex=False).str.strip()

df["commune_key"] = commune_key(df.home_commune)
canon = (df.groupby("commune_key").home_commune
         .agg(lambda s: s[s != s.str.upper()].mode().iloc[0]))   # most frequent mixed-case spelling
df["home_commune"] = df.commune_key.map(canon)
note(f"[Cleaning] raw municipality labels: {raw.home_commune.nunique()}, after harmonisation: "
     f"{df.home_commune.nunique()}")
moved = df.groupby("employee_id").home_commune.nunique().gt(1).sum()
note(f"[Cleaning] employees changing municipality between 2024 and 2025 after harmonisation: {moved}")

# 0.2 Missing income: imputed from the same employee's other year (income is identical across years for every
# employee observed twice), otherwise median full-time-equivalent income x fte
inc_varies = (df.dropna(subset=["gross_income_chf"]).groupby("employee_id").gross_income_chf.nunique() > 1).sum()
note(f"[Cleaning] employees whose observed income differs between years: {inc_varies}")
n_missing = df.gross_income_chf.isna().sum()
df["income_imputed"] = df.gross_income_chf.isna().astype(int)
df["gross_income_chf"] = df.groupby("employee_id").gross_income_chf.transform(lambda s: s.fillna(s.mean()))
still = df.gross_income_chf.isna()
med_fte_income = (df.gross_income_chf / df.fte).median()
df.loc[still, "gross_income_chf"] = (med_fte_income * df.loc[still, "fte"]).round(-2)
note(f"[Cleaning] missing income: {n_missing} rows; imputed from the other year: {n_missing - still.sum()}; "
     f"imputed with the full-time-equivalent median: {still.sum()} rows ({still.sum() // 2} employees)")

# 0.3 Car days out of range (E1001 in 2025: 7.4 > 5): capped at 5
bad = df.car_days_per_week > 5
note(f"[Cleaning] car_days_per_week > 5: {df.loc[bad, ['employee_id', 'year', 'car_days_per_week']].values.tolist()} "
     "-> capped at 5")
df["car_days_raw"] = df.car_days_per_week
df.loc[bad, "car_days_per_week"] = 5.0

# 0.4 Consistency checks
assert (df.uses_car == (df.main_mode == "car")).all()
assert ((df.uses_car == 1) == (df.car_days_per_week > 0)).all()
assert not df.duplicated(["employee_id", "year"]).any()
for c in ["site", "distance_km", "age", "gender", "n_children", "fte", "home_postcode"]:
    assert df.groupby("employee_id")[c].nunique().max() == 1, c
note("[Cleaning] age identical in 2024 and 2025 for every employee (age measured once)")

# 0.5 Derived variables
df["post"] = (df.year == 2025).astype(int)
df["lausanne"] = (df.site == "Lausanne").astype(int)
df["log_fee"] = np.log(df.parking_fee_chf)
df["income_fte"] = df.gross_income_chf / df.fte
emp = df[df.year == 2024].set_index("employee_id")
terc = pd.qcut(emp.income_fte, 3, labels=["T1 (low)", "T2 (middle)", "T3 (high)"])
df["income_tercile"] = df.employee_id.map(terc)
cuts = emp.income_fte.quantile([1 / 3, 2 / 3]).round(-2)
note(f"[Groups] full-time-equivalent income tercile cut-offs: {chf(cuts.iloc[0])} and {chf(cuts.iloc[1])} CHF")
df["children_group"] = df.n_children.map({0: "0 children", 1: "1 child", 2: "2 or more children"})
df["income_c"] = (df.income_fte - emp.income_fte.mean()) / 10_000   # centred, in CHF 10,000
df["log_dist"] = np.log(df.distance_km)
df["female"] = (df.gender == "F").astype(int)
df["age_c"] = df.age - emp.age.mean()


# =============================================================================
# Part A: public transport accessibility index
# =============================================================================
ta = pd.read_csv(ROOT / "transit_access.csv")
first = ta.first_arrival.str[:2].astype(int) * 60 + ta.first_arrival.str[3:].astype(int)
z = lambda x: (x - x.mean()) / x.std()
W_RATIO, W_DEP, W_FIRST = 0.4, 0.4, 0.2   # weights: arrival around 08:00 = departures 06:00-09:00 > first arrival
score = W_RATIO * z(-np.log(ta.pt_car_ratio)) + W_DEP * z(np.log(ta.departures_06_09)) + W_FIRST * z(-first)
score = 100 * (score - score.min()) / (score.max() - score.min())   # scale 0 (worst) to 100 (best)
assert np.allclose(score.round(1), ta.transit_score, atol=0.051), "recomputed score differs from the file"
# classes: score terciles computed separately for each site (Lausanne municipalities among themselves, Sion likewise)
cls = score.groupby(ta.site).transform(lambda x: pd.qcut(x, 3, labels=["Poor", "Medium", "Good"]).astype(str))
assert (cls == ta.transit_class).all()

ta["commune_key"] = commune_key(ta.home_commune)
assert set(df.commune_key) == set(ta.commune_key), "municipalities without an index"
df = df.merge(ta[["site", "commune_key", "transit_score", "transit_class", "car_km_osrm"]],
              on=["site", "commune_key"], how="left", validate="many_to_one")
assert df.transit_class.notna().all()
ORDER_T = ["Good", "Medium", "Poor"]
df["transit_class"] = pd.Categorical(df.transit_class, ORDER_T, ordered=True)

n_emp = df[df.year == 2024].groupby("home_commune").size()
tA1 = ta.assign(n_employes=ta.home_commune.map(n_emp))
tA1 = tA1[["site", "home_commune", "n_employes", "pt_time_min", "car_time_min", "pt_car_ratio",
           "departures_06_09", "first_arrival", "transit_score", "transit_class"]]
tA1.columns = ["Site", "Municipality", "Employees", "PT time (min)", "Car time (min)", "PT/car ratio",
               "Departures 06:00-09:00", "First arrival", "Score", "Class"]
tA1["Score"] = tA1.Score.round(1)
tA1["Car time (min)"] = tA1["Car time (min)"].map("{:.1f}".format)
tA1["PT/car ratio"] = tA1["PT/car ratio"].map("{:.2f}".format)
# municipality names with accents, for display only
ACCENTS = {"Chavannes-pres-Renens": "Chavannes-près-Renens", "Mezieres": "Mézières", "Vetroz": "Vétroz",
           "Savieze": "Savièse", "Heremence": "Hérémence", "Evolene": "Evolène"}
tA1["Municipality"] = tA1.Municipality.replace(ACCENTS)
save(tA1, "table_A1_communes")   # full list of the 39 municipalities (reference; also in transit_access.csv)

# Table A1 (report): compact summary by site and class, every municipality named
ta_d = ta.assign(first_min=first, Municipality=ta.home_commune.replace(ACCENTS),
                 n_employes=ta.home_commune.map(n_emp))
rows = []
for site in ["Lausanne", "Sion"]:
    for c in ORDER_T:
        g = ta_d[(ta_d.site == site) & (ta_d.transit_class == c)].sort_values("transit_score", ascending=False)
        fm = int(round(g.first_min.median()))
        rows.append({"Site": site, "Class": c, "Municipalities": len(g), "Employees": int(g.n_employes.sum()),
                     "PT/car ratio (median)": f"{g.pt_car_ratio.median():.2f}",
                     "Departures 06:00-09:00 (median)": f"{g.departures_06_09.median():.1f}",
                     "First arrival (median)": f"{fm // 60:02d}:{fm % 60:02d}",
                     "Score (min-max)": f"{g.transit_score.min():.1f}-{g.transit_score.max():.1f}",
                     "Municipalities (by decreasing score)": ", ".join(g.Municipality)})
tA1_summary = pd.DataFrame(rows)
save(tA1_summary, "table_A1")

# Validation: correlations with reported distance and class ordering
dist_c = df[df.year == 2024].groupby(["site", "commune_key"]).distance_km.mean()
v = ta.set_index(["site", "commune_key"]).join(dist_c)
note(f"[A] correlation OSRM distance vs mean reported distance by municipality: "
     f"{v.car_km_osrm.corr(v.distance_km):.2f}")
note(f"[A] correlation transit score vs mean reported distance: {v.transit_score.corr(v.distance_km):.2f}")
note("[A] employees by class and site:\n" +
     pd.crosstab(df[df.year == 2024].site, df[df.year == 2024].transit_class).to_string())


# =============================================================================
# Part B: extensive margin (uses_car)
# =============================================================================
LS = df[df.site == "Lausanne"]
SI = df[df.site == "Sion"]
DLOGFEE = np.log(FEE_2025 / FEE_2024)

# --- B1: mode shares and flows
rows = []
for site, g in df.groupby("site"):
    for y, gy in g.groupby("year"):
        sh = gy.main_mode.value_counts(normalize=True)
        rows.append({"Site": site, "Year": y, "Fee (CHF/day)": gy.parking_fee_chf.iloc[0], "N": len(gy),
                     "Car (%)": pct(sh.get("car", 0)), "Bus (%)": pct(sh.get("bus", 0)),
                     "Bike (%)": pct(sh.get("bike", 0))})
tB1 = pd.DataFrame(rows)

w = df.pivot(index="employee_id", columns="year", values="uses_car")
w["site"] = df[df.year == 2024].set_index("employee_id").site
flows = w.groupby("site").apply(lambda g: pd.Series({
    "Car -> other": int(((g[2024] == 1) & (g[2025] == 0)).sum()),
    "Other -> car": int(((g[2024] == 0) & (g[2025] == 1)).sum()),
}), include_groups=False)
flows["Net change"] = flows["Other -> car"] - flows["Car -> other"]
for site in ["Lausanne", "Sion"]:
    s0 = df[(df.site == site) & (df.year == 2024)].uses_car.mean()
    s1 = df[(df.site == site) & (df.year == 2025)].uses_car.mean()
    flows.loc[site, "Change in car share (pp)"] = f"{100 * (s1 - s0):.1f}"
    flows.loc[site, "Change in car share (%)"] = f"{100 * (s1 / s0 - 1):.1f}"
tB1 = tB1.merge(flows.astype(str).reset_index().rename(columns={"site": "Site"}), on="Site")
tB1.loc[tB1["Year"] == 2024, flows.columns] = ""
save(tB1, "table_B1")

# 2025 mode of Lausanne drivers who leave the car
mw = LS.pivot(index="employee_id", columns="year", values="main_mode")
note("[B1] Lausanne, 2025 mode of 2024 drivers who leave the car:",
     mw[(mw[2024] == "car") & (mw[2025] != "car")][2025].value_counts().to_dict())


def ext_stats(g):
    s0 = g[g.year == 2024].uses_car.mean()
    s1 = g[g.year == 2025].uses_car.mean()
    return s0, s1, 100 * (s1 - s0), np.log(s1 / s0) / DLOGFEE


s0, s1, dpp, e_ext = ext_stats(LS)
s0S, s1S, dppS, _ = ext_stats(SI)
note(f"[B] Lausanne: car share {pct(s0)} % -> {pct(s1)} %; {dpp:.1f} pp; extensive elasticity {e_ext:.2f}")
note(f"[B] Sion: car share {pct(s0S)} % -> {pct(s1S)} %; {dppS:.1f} pp")
note(f"[B] Lausanne net of Sion: {dpp - dppS:.1f} pp")

# --- B2: heterogeneity (Lausanne)
GROUPS = [("Transit access", "transit_class", ORDER_T),
          ("Income tercile (FTE)", "income_tercile", ["T1 (low)", "T2 (middle)", "T3 (high)"]),
          ("Children < 16", "children_group", ["0 children", "1 child", "2 or more children"])]


def by_group(data, fun, extra_total=True):
    out = []
    if extra_total:
        out.append(("Overall", "All", data))
    for title, var, levels in GROUPS:
        for lev in levels:
            out.append((title, lev, data[data[var] == lev]))
    return [(t, l, fun(g)) for t, l, g in out]


rows = []
for t, l, (a, b, d_, e) in by_group(LS, ext_stats):
    n = LS[(LS.year == 2024)] if t == "Overall" else None
    rows.append({"Dimension": t, "Group": l, "Car share 2024 (%)": pct(a), "Car share 2025 (%)": pct(b),
                 "Change (pp)": f"{d_:.1f}", "Extensive elasticity": f"{e:.2f}"})
tB2 = pd.DataFrame(rows)
tB2.insert(2, "N employees", [len(g) // 2 for _, _, g in
                             [("", "", LS)] + [(0, 0, LS[LS[v] == lev]) for _, v, lv in GROUPS for lev in lv]])
save(tB2, "table_B2")


# --- B3: linear probability model
def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data.employee_id})


CTRL = ("C(transit_class, Treatment('Good')) + income_c + C(n_children) + log_dist + age_c + female + C(fte)")
INTER = ("log_fee:C(transit_class, Treatment('Good')) + log_fee:income_c + log_fee:C(n_children)")


def four_specs(dep, data_all):
    lau = data_all[data_all.site == "Lausanne"]
    return {
        "(a) Lausanne": fit(f"{dep} ~ log_fee", lau),
        "(b) + Sion": fit(f"{dep} ~ log_fee + lausanne + post", data_all),
        "(c) + controls": fit(f"{dep} ~ log_fee + lausanne + post + {CTRL}", data_all),
        "(d) + interactions": fit(f"{dep} ~ log_fee + lausanne + post + {CTRL} + {INTER}", data_all),
    }


LABELS = [
    ("log_fee", "log(fee)"),
    ("log_fee:C(transit_class, Treatment('Good'))[T.Medium]", "log(fee) x medium transit access"),
    ("log_fee:C(transit_class, Treatment('Good'))[T.Poor]", "log(fee) x poor transit access"),
    ("log_fee:income_c", "log(fee) x FTE income (CHF 10,000, centred)"),
    ("log_fee:C(n_children)[T.1]", "log(fee) x 1 child"),
    ("log_fee:C(n_children)[T.2]", "log(fee) x 2 or more children"),
    ("lausanne", "Lausanne site"),
    ("post", "Year 2025"),
    ("C(transit_class, Treatment('Good'))[T.Medium]", "Medium transit access"),
    ("C(transit_class, Treatment('Good'))[T.Poor]", "Poor transit access"),
    ("income_c", "FTE income (CHF 10,000, centred)"),
    ("C(n_children)[T.1]", "1 child"),
    ("C(n_children)[T.2]", "2 or more children"),
    ("log_dist", "log(distance km)"),
    ("age_c", "Age (centred, years)"),
    ("female", "Female"),
    ("C(fte)[T.0.8]", "Workload (FTE) 0.8"),
    ("C(fte)[T.1.0]", "Workload (FTE) 1.0"),
    ("Intercept", "Constant"),
]


def reg_table(models, extra_rows):
    rows = []
    for key, lab in LABELS:
        if not any(key in m.params for m in models.values()):
            continue
        cr, sr = {"Variable": lab}, {"Variable": ""}
        for name, m in models.items():
            if key in m.params:
                cr[name] = f"{m.params[key]:.3f}{stars(m.pvalues[key])}"
                sr[name] = f"({m.bse[key]:.3f})"
            else:
                cr[name] = sr[name] = ""
        rows += [cr, sr]
    for lab, fun in extra_rows:
        rows.append({"Variable": lab, **{n: fun(m) for n, m in models.items()}})
    rows.append({"Variable": "Observations", **{n: f"{int(m.nobs)}" for n, m in models.items()}})
    rows.append({"Variable": "Employees (clusters)",
                 **{n: f"{m.model.data.frame.employee_id.nunique()}" for n, m in models.items()}})
    rows.append({"Variable": "R2", **{n: f"{m.rsquared:.3f}" for n, m in models.items()}})
    return pd.DataFrame(rows)


mB = four_specs("uses_car", df)
tB3 = reg_table(mB, [("Effect of doubling the fee (pp)",
                      lambda m: f"{100 * m.params['log_fee'] * np.log(2):.1f}")])
save(tB3, "table_B3")
for n, m in mB.items():
    b, se = m.params["log_fee"], m.bse["log_fee"]
    note(f"[B3] {n}: coef log(fee) = {b:.3f} (se {se:.3f}); doubling = {100 * b * np.log(2):.1f} pp "
         f"[95% CI {100 * (b - 1.96 * se) * np.log(2):.1f}; {100 * (b + 1.96 * se) * np.log(2):.1f}]")


# =============================================================================
# Part C: intensive margin (drivers) and decomposition
# =============================================================================
CU = df[df.uses_car == 1].copy()
CU["log_days"] = np.log(CU.car_days_per_week)


def int_stats(g):
    c = g[g.uses_car == 1]
    d0 = c[c.year == 2024].car_days_per_week.mean()
    d1 = c[c.year == 2025].car_days_per_week.mean()
    return d0, d1, np.log(d1 / d0) / DLOGFEE, (c.year == 2024).sum(), (c.year == 2025).sum()


rows = []
for t, l, (d0, d1, e, n0, n1) in by_group(LS, int_stats):
    rows.append({"Dimension": t, "Group": l, "Days/week 2024": f"{d0:.2f}", "Days/week 2025": f"{d1:.2f}",
                 "Intensive elasticity": f"{e:.2f}", "N drivers 2024": n0, "N drivers 2025": n1})
tC1 = pd.DataFrame(rows)
save(tC1, "table_C1")
d0, d1, e_int, _, _ = int_stats(LS)
d0S, d1S, _, _, _ = int_stats(SI)
note(f"[C] Lausanne: days per driver {d0:.2f} -> {d1:.2f}; intensive elasticity {e_int:.2f}")
note(f"[C] Sion: days per driver {d0S:.2f} -> {d1S:.2f}")

mC = four_specs("log_days", CU)
tC2 = reg_table(mC, [("Intensive elasticity (= coef. on log(fee))", lambda m: f"{m.params['log_fee']:.2f}"),
                     ("Effect of doubling the fee (%)",
                      lambda m: f"{100 * (2 ** m.params['log_fee'] - 1):.1f}")])
save(tC2, "table_C2")
for n, m in mC.items():
    note(f"[C2] {n}: elasticity {m.params['log_fee']:.3f} (se {m.bse['log_fee']:.3f})")

# Selection: 2024 vs 2025 drivers in Lausanne, and stayers vs leavers
w_ls = LS.pivot(index="employee_id", columns="year", values="uses_car")
e24 = LS[LS.year == 2024].set_index("employee_id")
stay = w_ls[(w_ls[2024] == 1) & (w_ls[2025] == 1)].index
leave = w_ls[(w_ls[2024] == 1) & (w_ls[2025] == 0)].index
enter = w_ls[(w_ls[2024] == 0) & (w_ls[2025] == 1)].index


def profile(ix):
    g = e24.loc[ix]
    return pd.Series({"N": len(g), "Days/week 2024": g.car_days_per_week.mean(),
                      "Distance (km)": g.distance_km.mean(), "Poor transit access (%)": 100 * (g.transit_class == "Poor").mean(),
                      "Good transit access (%)": 100 * (g.transit_class == "Good").mean(),
                      "FTE income (CHF)": g.income_fte.mean(), "Children (mean)": g.n_children.mean()})


cu24 = LS[(LS.year == 2024) & (LS.uses_car == 1)].employee_id
cu25 = LS[(LS.year == 2025) & (LS.uses_car == 1)].employee_id
sel = pd.DataFrame({"Drivers 2024": profile(cu24), "Drivers 2025": profile(cu25),
                    "Stay": profile(stay), "Leave the car": profile(leave)}).round(2)
save(sel, "table_C_selection", index=True)
note("[C] selection (2024 characteristics, Lausanne):\n" + sel.to_string())

# Within-person intensive elasticity (drivers in both years), employee fixed effects
stayers = CU[CU.employee_id.isin(CU.groupby("employee_id").size().loc[lambda s: s == 2].index)]
m_fe = fit("log_days ~ log_fee + post + C(employee_id)", stayers)
note(f"[C] intensive elasticity, drivers in both years, employee + year fixed effects (Lausanne vs Sion): "
     f"{m_fe.params['log_fee']:.3f} (se {m_fe.bse['log_fee']:.3f}), N = {int(m_fe.nobs)}")
# variant without capping E1001: drop the row
m_drop = fit(f"log_days ~ log_fee + lausanne + post + {CTRL}", CU[CU.car_days_raw <= 5])
note(f"[C] specification (c) dropping E1001-2025 instead of capping: {m_drop.params['log_fee']:.3f}")


# --- C3: decomposition  ln(T) = ln(S) + ln(D),  T = car days per employee
def decomp(g):
    out = {}
    for y in (2024, 2025):
        gy = g[g.year == y]
        S = gy.uses_car.mean()
        D = gy[gy.uses_car == 1].car_days_per_week.mean()
        T = gy.car_days_per_week.mean()
        out[y] = (S, D, T)
    return out


def elast(dL, dS=None):
    """elasticities (ext, int, total); if dS, net of the change in Sion (unchanged fee)."""
    e = [np.log(dL[2025][i] / dL[2024][i]) for i in range(3)]
    if dS is not None:
        e = [e[i] - np.log(dS[2025][i] / dS[2024][i]) for i in range(3)]
    return np.array(e) / DLOGFEE


dL, dS = decomp(LS), decomp(SI)
for y in (2024, 2025):
    assert np.isclose(dL[y][0] * dL[y][1], dL[y][2])
e_raw, e_did = elast(dL), elast(dL, dS)


# Cluster bootstrap (employees, drawn within each site)
def boot():
    ids_L = LS.employee_id.unique()
    ids_S = SI.employee_id.unique()
    by_id = {k: g for k, g in df.groupby("employee_id")}
    arrL = {k: by_id[k][["year", "uses_car", "car_days_per_week"]].to_numpy() for k in ids_L}
    arrS = {k: by_id[k][["year", "uses_car", "car_days_per_week"]].to_numpy() for k in ids_S}

    def dec(arr):
        out = {}
        for y in (2024, 2025):
            a = arr[arr[:, 0] == y]
            S = a[:, 1].mean()
            T = a[:, 2].mean()
            out[y] = (S, T / S, T)
        return out

    stackL = np.stack([arrL[k] for k in ids_L])
    stackS = np.stack([arrS[k] for k in ids_S])
    res_raw, res_did = [], []
    for _ in range(N_BOOT):
        bL = stackL[RNG.integers(0, len(ids_L), len(ids_L))].reshape(-1, 3)
        bS = stackS[RNG.integers(0, len(ids_S), len(ids_S))].reshape(-1, 3)
        dl, ds = dec(bL), dec(bS)
        res_raw.append(elast(dl))
        res_did.append(elast(dl, ds))
    return np.array(res_raw), np.array(res_did)


b_raw, b_did = boot()
ci = lambda b: np.percentile(b, [2.5, 97.5], axis=0)
ci_raw, ci_did = ci(b_raw), ci(b_did)

rowsC3 = []
for lab, i in [("Extensive: share of drivers S", 0), ("Intensive: days per driver D", 1),
               ("Total: car days per employee T = S x D", 2)]:
    fmt = (lambda v: f"{100 * v:.1f} %") if i == 0 else (lambda v: f"{v:.2f}")
    rowsC3.append({
        "Margin": lab,
        "Lausanne 2024": fmt(dL[2024][i]), "Lausanne 2025": fmt(dL[2025][i]),
        "Sion 2024": fmt(dS[2024][i]), "Sion 2025": fmt(dS[2025][i]),
        "Elasticity, Lausanne before-after": f"{e_raw[i]:.2f} [{ci_raw[0, i]:.2f}; {ci_raw[1, i]:.2f}]",
        "Elasticity, net of Sion": f"{e_did[i]:.2f} [{ci_did[0, i]:.2f}; {ci_did[1, i]:.2f}]",
    })
tC3 = pd.DataFrame(rowsC3)
save(tC3, "table_C3")
note(f"[C3] before-after: ext {e_raw[0]:.3f} + int {e_raw[1]:.3f} = {e_raw[0] + e_raw[1]:.3f}; total {e_raw[2]:.3f}")
note(f"[C3] net of Sion: ext {e_did[0]:.3f} + int {e_did[1]:.3f} = {e_did[0] + e_did[1]:.3f}; total {e_did[2]:.3f}")
note(f"[C3] share of the extensive margin in the total (net of Sion): {100 * e_did[0] / e_did[2]:.0f} %")


# =============================================================================
# Part D: 2026 scenarios (Lausanne site, 800 employees)
# =============================================================================
N_LAU = LS.employee_id.nunique()
S4, D4 = dL[2025][0], dL[2025][1]


def scenario(eps_ext, eps_int, fees=SCENARIOS):
    """Constant elasticities applied from the observed CHF 4 baseline: X(p) = X(4) * (p/4)^eps."""
    rows = []
    for p in fees:
        S = S4 * (p / FEE_2025) ** eps_ext
        D = D4 * (p / FEE_2025) ** eps_int
        T = S * D
        rev = p * T * WEEKS * N_LAU
        rows.append({"fee": p, "S": S, "D": D, "T": T, "rev": rev})
    return pd.DataFrame(rows)


central = scenario(e_did[0], e_did[1])
alt_raw = scenario(e_raw[0], e_raw[1])
lo_ext, hi_ext = ci_did[:, 0]       # sensitivity: 95% CI bounds of the extensive elasticity
lo_int, hi_int = ci_did[:, 1]
alt_lo = scenario(lo_ext, lo_int)   # strong response
alt_hi = scenario(hi_ext, hi_int)   # weak response

# Linear variant from LPM (c): S(p) = S4 + a * ln(p/4)
a_c = mB["(c) + controls"].params["log_fee"]
lpm = central.copy()
lpm["S"] = S4 + a_c * np.log(lpm.fee / FEE_2025)
lpm["T"] = lpm.S * lpm.D
lpm["rev"] = lpm.fee * lpm["T"] * WEEKS * N_LAU

rev_2025_obs = (LS[LS.year == 2025].car_days_per_week.sum() * WEEKS * FEE_2025)
assert np.isclose(central.rev.iloc[0], rev_2025_obs)

tD1 = pd.DataFrame({
    "Fee (CHF/day)": central.fee.map(lambda p: f"{p} (2025 baseline)" if p == 4 else f"{p}"),
    "Car share (%)": central.S.map(pct),
    "Days per driver": central.D.map(lambda v: f"{v:.2f}"),
    "Car days per employee": central["T"].map(lambda v: f"{v:.2f}"),
    "Annual revenue (thousand CHF)": central.rev.map(lambda v: f"{v / 1000:.0f}"),
    "Sensitivity: car share, 95% CI bounds (%)": [f"{pct(l)} to {pct(h)}" if p != 4 else "" for p, l, h in
                                             zip(central.fee, alt_lo.S, alt_hi.S)],
    "Sensitivity: before-after without Sion, car share (%)": alt_raw.S.map(pct),
    "Sensitivity: linear LPM (c), car share (%)": lpm.S.map(pct),
    "Sensitivity: revenue, 95% CI bounds (thousand CHF)": [f"{l / 1000:.0f} to {h / 1000:.0f}" if p != 4 else "" for p, l, h in
                                                  zip(central.fee, alt_lo.rev, alt_hi.rev)],
})
save(tD1, "table_D1")
note(f"[D] central elasticities (net of Sion): extensive {e_did[0]:.3f}, intensive {e_did[1]:.3f}")
note("[D] central scenario:\n" + central.round(4).to_string())
note(f"[D] observed 2024 revenue, Lausanne: {chf(LS[LS.year == 2024].car_days_per_week.sum() * WEEKS * FEE_2024)} CHF")

# --- Figure D1: two panels (no dual axis)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
x = central.fee
for ax, col, scale, ylab, title in [
        (axes[0], "S", 100, "Employees commuting by car (%)", "Car share"),
        (axes[1], "rev", 1e-3, "Annual revenue (thousand CHF)", "Parking revenue")]:
    ax.fill_between(x, alt_lo[col] * scale, alt_hi[col] * scale, color=BLUE, alpha=0.15, lw=0,
                    label="95% CI bounds (ext. and int.)")
    ax.plot(x, central[col] * scale, color=BLUE, lw=2, marker="o", ms=4, label="Central (net of Sion)")
    ax.plot(x, alt_raw[col] * scale, color=ORANGE, lw=2, ls="--", marker="s", ms=4,
            label="Lausanne before-after")
    ax.axvspan(8, 10.5, color="#efefec", zorder=0)
    ax.set_xlim(3.7, 10.3)
    ax.set_xticks(SCENARIOS)
    ax.set_xlabel("Daily fee (CHF)")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e2e1dc", lw=0.6)
    ax.text(8.1, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()), "beyond a second\ndoubling",
            fontsize=7, color=MUTED, va="bottom")
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "figure_D1.pdf")
fig.savefig(FIG / "figure_D1.png", dpi=200)
plt.close(fig)


# =============================================================================
# Part E: incidence (annual parking spending as % of observed gross income)
# =============================================================================
E = df[df.year == 2025].copy()
E["spend"] = E.parking_fee_chf * E.car_days_per_week * WEEKS
E["burden"] = 100 * E.spend / E.gross_income_chf
# projection at the recommended fee (Lausanne only): elasticities specific to each transit access class
# (Lausanne before-after, Tables B2 and C1), since transit access is the dimension where the response differs most;
# each 2025 driver still drives with probability (p/4)^eps_ext and scales days by (p/4)^eps_int
ratio = RECO_FEE / FEE_2025
eps_cls = {c: (ext_stats(LS[LS.transit_class == c])[3], int_stats(LS[LS.transit_class == c])[2]) for c in ORDER_T}
note("[E] elasticities (ext, int) by transit class used for the projection:",
     {c: (round(a, 3), round(b, 3)) for c, (a, b) in eps_cls.items()})
e_ext_i = E.transit_class.map(lambda c: eps_cls[c][0]).astype(float)
e_int_i = E.transit_class.map(lambda c: eps_cls[c][1]).astype(float)
E["burden_reco_driver"] = np.where(E.uses_car == 1, 100 * RECO_FEE * E.car_days_per_week * ratio ** e_int_i
                                   * WEEKS / E.gross_income_chf, np.nan)
E["burden_reco_all"] = np.where(E.uses_car == 1, E.burden_reco_driver * ratio ** e_ext_i, 0.0)

EL = E[E.site == "Lausanne"]
rows = []
for t, l, g in [("Overall", "All", EL)] + [(t, lev, EL[EL[v] == lev]) for t, v, lv in GROUPS for lev in lv]:
    dr = g[g.uses_car == 1]
    rows.append({"Dimension": t, "Group": l, "N employees": len(g), "N drivers 2025": len(dr),
                 "(a) All employees, 2025 (%)": f"{g.burden.mean():.2f}",
                 "(b) Drivers 2025 (%)": f"{dr.burden.mean():.2f}",
                 f"(b) Mean spending of 2025 drivers (CHF/year)": f"{dr.spend.mean():.0f}",
                 f"(c) All, at CHF {RECO_FEE} (%)": f"{g.burden_reco_all.mean():.2f}",
                 f"(c) Remaining drivers, at CHF {RECO_FEE} (%)": f"{dr.burden_reco_driver.mean():.2f}",
                 "Share of 2025 revenue (%)": pct(g.spend.sum() / EL.spend.sum())})
ES = E[E.site == "Sion"]
drS = ES[ES.uses_car == 1]
rows.append({"Dimension": "Reference", "Group": "Sion (CHF 2)", "N employees": len(ES),
             "N drivers 2025": len(drS), "(a) All employees, 2025 (%)": f"{ES.burden.mean():.2f}",
             "(b) Drivers 2025 (%)": f"{drS.burden.mean():.2f}",
             "(b) Mean spending of 2025 drivers (CHF/year)": f"{drS.spend.mean():.0f}",
             f"(c) All, at CHF {RECO_FEE} (%)": "", f"(c) Remaining drivers, at CHF {RECO_FEE} (%)": "",
             "Share of 2025 revenue (%)": ""})
tE1 = pd.DataFrame(rows)
save(tE1, "table_E1")
note(f"[E] maximum individual burden 2025 (Lausanne): {EL.burden.max():.2f} %; "
     f"share of drivers above 1 %: {pct((EL[EL.uses_car == 1].burden > 1).mean())} %")

(OUT / "results.txt").write_text("\n".join(LOG) + "\n")
print(f"\nTables written to {TAB}, figure to {FIG}")
