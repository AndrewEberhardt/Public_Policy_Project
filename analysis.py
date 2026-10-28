"""Tarification du parking et mode de transport : script d'analyse reproductible.

Entrées : parking_data.csv, transit_access.csv (même dossier que ce script).
Sorties : output/tables/*.csv et output/tables/*.md (tableaux A1, B1-B3, C1-C3, D1, E1),
          output/figures/figure_D1.pdf et .png, output/results.txt (chiffres cités dans le texte).

Exécution : python analysis.py

Conventions d'arrondi (identiques dans tout le rapport)
- parts et taux en %         : 1 décimale
- variations en points de %  : 1 décimale
- élasticités                : 2 décimales
- coefficients de régression : 3 décimales (erreurs standard entre parenthèses)
- jours par semaine          : 2 décimales
- montants en CHF            : arrondis à l'unité (recettes au millier de CHF)
- taux d'effort (% du revenu): 2 décimales
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

WEEKS = 46                      # semaines travaillées par an
FEE_2024, FEE_2025 = 2.0, 4.0   # tarif Lausanne
SCENARIOS = [4, 5, 6, 8, 10]    # CHF par jour, 4 = base 2025
RECO_FEE = 6                    # tarif recommandé pour 2026 (Partie F)
N_BOOT = 999
RNG = np.random.default_rng(2026)

LOG = []


def note(*args):
    """Enregistre un résultat cité dans le texte."""
    line = " ".join(str(a) for a in args)
    LOG.append(line)
    print(line)


def save(df, name, index=False):
    df.to_csv(TAB / f"{name}.csv", index=index)
    d = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * d.shape[1]]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False)]
    (TAB / f"{name}.md").write_text("\n".join(lines) + "\n")


def pct(x):
    return f"{100 * x:.1f}"


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


# =============================================================================
# 0. Chargement et nettoyage
# =============================================================================
raw = pd.read_csv(ROOT / "parking_data.csv")
df = raw.copy()

# 0.1 Communes : 62 libellés -> 39 communes (casse et tirets incohérents)
def commune_key(s):
    return s.str.upper().str.replace("-", " ", regex=False).str.strip()

df["commune_key"] = commune_key(df.home_commune)
canon = (df.groupby("commune_key").home_commune
         .agg(lambda s: s[s != s.str.upper()].mode().iloc[0]))   # graphie en casse mixte la plus fréquente
df["home_commune"] = df.commune_key.map(canon)
note(f"[Nettoyage] libellés de commune bruts : {raw.home_commune.nunique()}, après harmonisation : "
     f"{df.home_commune.nunique()}")
moved = df.groupby("employee_id").home_commune.nunique().gt(1).sum()
note(f"[Nettoyage] employés changeant de commune entre 2024 et 2025 après harmonisation : {moved}")

# 0.2 Revenu manquant : imputation depuis l'autre année du même employé (revenu identique d'une année à l'autre
# pour tous les employés observés deux fois), sinon médiane du revenu équivalent plein temps x fte
inc_varies = (df.dropna(subset=["gross_income_chf"]).groupby("employee_id").gross_income_chf.nunique() > 1).sum()
note(f"[Nettoyage] employés dont le revenu observé diffère entre années : {inc_varies}")
n_missing = df.gross_income_chf.isna().sum()
df["income_imputed"] = df.gross_income_chf.isna().astype(int)
df["gross_income_chf"] = df.groupby("employee_id").gross_income_chf.transform(lambda s: s.fillna(s.mean()))
still = df.gross_income_chf.isna()
med_fte_income = (df.gross_income_chf / df.fte).median()
df.loc[still, "gross_income_chf"] = (med_fte_income * df.loc[still, "fte"]).round(-2)
note(f"[Nettoyage] revenus manquants : {n_missing} lignes ; imputés depuis l'autre année : {n_missing - still.sum()} ; "
     f"imputés par la médiane équivalent plein temps : {still.sum()} lignes ({still.sum() // 2} employés)")

# 0.3 Jours de voiture hors bornes (E1001 en 2025 : 7,4 > 5) : plafonné à 5
bad = df.car_days_per_week > 5
note(f"[Nettoyage] car_days_per_week > 5 : {df.loc[bad, ['employee_id', 'year', 'car_days_per_week']].values.tolist()} "
     "-> plafonné à 5")
df["car_days_raw"] = df.car_days_per_week
df.loc[bad, "car_days_per_week"] = 5.0

# 0.4 Contrôles de cohérence
assert (df.uses_car == (df.main_mode == "car")).all()
assert ((df.uses_car == 1) == (df.car_days_per_week > 0)).all()
assert not df.duplicated(["employee_id", "year"]).any()
for c in ["site", "distance_km", "age", "gender", "n_children", "fte", "home_postcode"]:
    assert df.groupby("employee_id")[c].nunique().max() == 1, c
note("[Nettoyage] âge identique en 2024 et 2025 pour tous les employés (âge mesuré une seule fois)")

# 0.5 Variables dérivées
df["post"] = (df.year == 2025).astype(int)
df["lausanne"] = (df.site == "Lausanne").astype(int)
df["log_fee"] = np.log(df.parking_fee_chf)
df["income_fte"] = df.gross_income_chf / df.fte
emp = df[df.year == 2024].set_index("employee_id")
terc = pd.qcut(emp.income_fte, 3, labels=["T1 (bas)", "T2 (moyen)", "T3 (haut)"])
df["income_tercile"] = df.employee_id.map(terc)
cuts = emp.income_fte.quantile([1 / 3, 2 / 3]).round(-2)
note(f"[Groupes] bornes des terciles de revenu équivalent plein temps : {cuts.iloc[0]:,.0f} et {cuts.iloc[1]:,.0f} CHF")
df["children_group"] = df.n_children.map({0: "0 enfant", 1: "1 enfant", 2: "2 enfants ou plus"})
df["income_c"] = (df.income_fte - emp.income_fte.mean()) / 10_000   # centré, en 10 000 CHF
df["log_dist"] = np.log(df.distance_km)
df["female"] = (df.gender == "F").astype(int)
df["age_c"] = df.age - emp.age.mean()


# =============================================================================
# Partie A : indice d'accès aux transports publics
# =============================================================================
ta = pd.read_csv(ROOT / "transit_access.csv")
first = ta.first_arrival.str[:2].astype(int) * 60 + ta.first_arrival.str[3:].astype(int)
z = lambda x: (x - x.mean()) / x.std()
score = (z(-np.log(ta.pt_car_ratio)) + z(np.log(ta.departures_06_09)) + z(-first)) / 3
assert np.allclose(score.round(3), ta.transit_score, atol=1e-3), "score recalculé différent du fichier"
cls = pd.qcut(score, 3, labels=["Faible", "Moyen", "Bon"]).astype(str)
assert (cls == ta.transit_class).all()

ta["commune_key"] = commune_key(ta.home_commune)
assert set(df.commune_key) == set(ta.commune_key), "communes sans indice"
df = df.merge(ta[["site", "commune_key", "transit_score", "transit_class", "car_km_osrm"]],
              on=["site", "commune_key"], how="left", validate="many_to_one")
assert df.transit_class.notna().all()
ORDER_T = ["Bon", "Moyen", "Faible"]
df["transit_class"] = pd.Categorical(df.transit_class, ORDER_T, ordered=True)

n_emp = df[df.year == 2024].groupby("home_commune").size()
tA1 = ta.assign(n_employes=ta.home_commune.map(n_emp))
tA1 = tA1[["site", "home_commune", "n_employes", "pt_time_min", "car_time_min", "pt_car_ratio",
           "departures_06_09", "first_arrival", "transit_score", "transit_class"]]
tA1.columns = ["Site", "Commune", "Employés", "Temps TP (min)", "Temps voiture (min)", "Ratio TP/voiture",
               "Départs 06h-09h", "Première arrivée", "Score", "Classe"]
tA1["Score"] = tA1.Score.round(2)
save(tA1, "table_A1")

# Validation : corrélations avec la distance déclarée et ordre des classes
dist_c = df[df.year == 2024].groupby(["site", "commune_key"]).distance_km.mean()
v = ta.set_index(["site", "commune_key"]).join(dist_c)
note(f"[A] corrélation distance OSRM vs distance moyenne déclarée par commune : "
     f"{v.car_km_osrm.corr(v.distance_km):.2f}")
note(f"[A] corrélation score TP vs distance moyenne déclarée : {v.transit_score.corr(v.distance_km):.2f}")
note("[A] employés par classe et site :\n" +
     pd.crosstab(df[df.year == 2024].site, df[df.year == 2024].transit_class).to_string())


# =============================================================================
# Partie B : marge extensive (uses_car)
# =============================================================================
LS = df[df.site == "Lausanne"]
SI = df[df.site == "Sion"]
DLOGFEE = np.log(FEE_2025 / FEE_2024)

# --- B1 : parts modales et flux
rows = []
for site, g in df.groupby("site"):
    for y, gy in g.groupby("year"):
        sh = gy.main_mode.value_counts(normalize=True)
        rows.append({"Site": site, "Année": y, "Tarif (CHF/jour)": gy.parking_fee_chf.iloc[0], "N": len(gy),
                     "Voiture (%)": pct(sh.get("car", 0)), "Bus (%)": pct(sh.get("bus", 0)),
                     "Vélo (%)": pct(sh.get("bike", 0))})
tB1 = pd.DataFrame(rows)

w = df.pivot(index="employee_id", columns="year", values="uses_car")
w["site"] = df[df.year == 2024].set_index("employee_id").site
flows = w.groupby("site").apply(lambda g: pd.Series({
    "Voiture -> autre": int(((g[2024] == 1) & (g[2025] == 0)).sum()),
    "Autre -> voiture": int(((g[2024] == 0) & (g[2025] == 1)).sum()),
}), include_groups=False)
flows["Variation nette"] = flows["Autre -> voiture"] - flows["Voiture -> autre"]
for site in ["Lausanne", "Sion"]:
    s0 = df[(df.site == site) & (df.year == 2024)].uses_car.mean()
    s1 = df[(df.site == site) & (df.year == 2025)].uses_car.mean()
    flows.loc[site, "Variation part voiture (pp)"] = f"{100 * (s1 - s0):.1f}"
    flows.loc[site, "Variation part voiture (%)"] = f"{100 * (s1 / s0 - 1):.1f}"
tB1 = tB1.merge(flows.astype(str).reset_index().rename(columns={"site": "Site"}), on="Site")
tB1.loc[tB1["Année"] == 2024, flows.columns] = ""
save(tB1, "table_B1")

# mode de destination des automobilistes de Lausanne qui quittent la voiture
mw = LS.pivot(index="employee_id", columns="year", values="main_mode")
note("[B1] Lausanne, mode 2025 des automobilistes 2024 qui quittent la voiture :",
     mw[(mw[2024] == "car") & (mw[2025] != "car")][2025].value_counts().to_dict())


def ext_stats(g):
    s0 = g[g.year == 2024].uses_car.mean()
    s1 = g[g.year == 2025].uses_car.mean()
    return s0, s1, 100 * (s1 - s0), np.log(s1 / s0) / DLOGFEE


s0, s1, dpp, e_ext = ext_stats(LS)
s0S, s1S, dppS, _ = ext_stats(SI)
note(f"[B] Lausanne : part voiture {pct(s0)} % -> {pct(s1)} % ; {dpp:.1f} pp ; élasticité extensive {e_ext:.2f}")
note(f"[B] Sion : part voiture {pct(s0S)} % -> {pct(s1S)} % ; {dppS:.1f} pp")
note(f"[B] Lausanne net de Sion : {dpp - dppS:.1f} pp")

# --- B2 : hétérogénéité (Lausanne)
GROUPS = [("Accès TP", "transit_class", ORDER_T),
          ("Tercile de revenu (EPT)", "income_tercile", ["T1 (bas)", "T2 (moyen)", "T3 (haut)"]),
          ("Enfants < 16 ans", "children_group", ["0 enfant", "1 enfant", "2 enfants ou plus"])]


def by_group(data, fun, extra_total=True):
    out = []
    if extra_total:
        out.append(("Ensemble", "Tous", data))
    for title, var, levels in GROUPS:
        for lev in levels:
            out.append((title, lev, data[data[var] == lev]))
    return [(t, l, fun(g)) for t, l, g in out]


rows = []
for t, l, (a, b, d_, e) in by_group(LS, ext_stats):
    n = LS[(LS.year == 2024)] if t == "Ensemble" else None
    rows.append({"Dimension": t, "Groupe": l, "Part voiture 2024 (%)": pct(a), "Part voiture 2025 (%)": pct(b),
                 "Variation (pp)": f"{d_:.1f}", "Élasticité extensive": f"{e:.2f}"})
tB2 = pd.DataFrame(rows)
tB2.insert(2, "N employés", [len(g) // 2 for _, _, g in
                             [("", "", LS)] + [(0, 0, LS[LS[v] == lev]) for _, v, lv in GROUPS for lev in lv]])
save(tB2, "table_B2")


# --- B3 : modèle de probabilité linéaire
def fit(formula, data):
    return smf.ols(formula, data=data).fit(cov_type="cluster", cov_kwds={"groups": data.employee_id})


CTRL = ("C(transit_class, Treatment('Bon')) + income_c + C(n_children) + log_dist + age_c + female + C(fte)")
INTER = ("log_fee:C(transit_class, Treatment('Bon')) + log_fee:income_c + log_fee:C(n_children)")


def four_specs(dep, data_all):
    lau = data_all[data_all.site == "Lausanne"]
    return {
        "(a) Lausanne": fit(f"{dep} ~ log_fee", lau),
        "(b) + Sion": fit(f"{dep} ~ log_fee + lausanne + post", data_all),
        "(c) + contrôles": fit(f"{dep} ~ log_fee + lausanne + post + {CTRL}", data_all),
        "(d) + interactions": fit(f"{dep} ~ log_fee + lausanne + post + {CTRL} + {INTER}", data_all),
    }


LABELS = [
    ("log_fee", "log(tarif)"),
    ("log_fee:C(transit_class, Treatment('Bon'))[T.Moyen]", "log(tarif) x accès TP moyen"),
    ("log_fee:C(transit_class, Treatment('Bon'))[T.Faible]", "log(tarif) x accès TP faible"),
    ("log_fee:income_c", "log(tarif) x revenu EPT (10 000 CHF, centré)"),
    ("log_fee:C(n_children)[T.1]", "log(tarif) x 1 enfant"),
    ("log_fee:C(n_children)[T.2]", "log(tarif) x 2 enfants ou plus"),
    ("lausanne", "Site Lausanne"),
    ("post", "Année 2025"),
    ("C(transit_class, Treatment('Bon'))[T.Moyen]", "Accès TP moyen"),
    ("C(transit_class, Treatment('Bon'))[T.Faible]", "Accès TP faible"),
    ("income_c", "Revenu EPT (10 000 CHF, centré)"),
    ("C(n_children)[T.1]", "1 enfant"),
    ("C(n_children)[T.2]", "2 enfants ou plus"),
    ("log_dist", "log(distance km)"),
    ("age_c", "Âge (centré, années)"),
    ("female", "Femme"),
    ("C(fte)[T.0.8]", "Taux d'activité 0,8"),
    ("C(fte)[T.1.0]", "Taux d'activité 1,0"),
    ("Intercept", "Constante"),
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
    rows.append({"Variable": "Employés (grappes)",
                 **{n: f"{m.model.data.frame.employee_id.nunique()}" for n, m in models.items()}})
    rows.append({"Variable": "R2", **{n: f"{m.rsquared:.3f}" for n, m in models.items()}})
    return pd.DataFrame(rows)


mB = four_specs("uses_car", df)
tB3 = reg_table(mB, [("Effet d'un doublement du tarif (pp)",
                      lambda m: f"{100 * m.params['log_fee'] * np.log(2):.1f}")])
save(tB3, "table_B3")
for n, m in mB.items():
    b, se = m.params["log_fee"], m.bse["log_fee"]
    note(f"[B3] {n}: coef log(tarif) = {b:.3f} (se {se:.3f}) ; doublement = {100 * b * np.log(2):.1f} pp "
         f"[IC95 {100 * (b - 1.96 * se) * np.log(2):.1f} ; {100 * (b + 1.96 * se) * np.log(2):.1f}]")


# =============================================================================
# Partie C : marge intensive (automobilistes) et décomposition
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
    rows.append({"Dimension": t, "Groupe": l, "Jours/sem. 2024": f"{d0:.2f}", "Jours/sem. 2025": f"{d1:.2f}",
                 "Élasticité intensive": f"{e:.2f}", "N automobilistes 2024": n0, "N automobilistes 2025": n1})
tC1 = pd.DataFrame(rows)
save(tC1, "table_C1")
d0, d1, e_int, _, _ = int_stats(LS)
d0S, d1S, _, _, _ = int_stats(SI)
note(f"[C] Lausanne : jours/automobiliste {d0:.2f} -> {d1:.2f} ; élasticité intensive {e_int:.2f}")
note(f"[C] Sion : jours/automobiliste {d0S:.2f} -> {d1S:.2f}")

mC = four_specs("log_days", CU)
tC2 = reg_table(mC, [("Élasticité intensive (= coef. log(tarif))", lambda m: f"{m.params['log_fee']:.2f}"),
                     ("Effet d'un doublement du tarif (%)",
                      lambda m: f"{100 * (2 ** m.params['log_fee'] - 1):.1f}")])
save(tC2, "table_C2")
for n, m in mC.items():
    note(f"[C2] {n}: élasticité {m.params['log_fee']:.3f} (se {m.bse['log_fee']:.3f})")

# Sélection : automobilistes 2024 vs 2025 à Lausanne, et restants vs sortants
w_ls = LS.pivot(index="employee_id", columns="year", values="uses_car")
e24 = LS[LS.year == 2024].set_index("employee_id")
stay = w_ls[(w_ls[2024] == 1) & (w_ls[2025] == 1)].index
leave = w_ls[(w_ls[2024] == 1) & (w_ls[2025] == 0)].index
enter = w_ls[(w_ls[2024] == 0) & (w_ls[2025] == 1)].index


def profile(ix):
    g = e24.loc[ix]
    return pd.Series({"N": len(g), "Jours/sem. 2024": g.car_days_per_week.mean(),
                      "Distance (km)": g.distance_km.mean(), "Accès TP faible (%)": 100 * (g.transit_class == "Faible").mean(),
                      "Accès TP bon (%)": 100 * (g.transit_class == "Bon").mean(),
                      "Revenu EPT (CHF)": g.income_fte.mean(), "Enfants (moy.)": g.n_children.mean()})


cu24 = LS[(LS.year == 2024) & (LS.uses_car == 1)].employee_id
cu25 = LS[(LS.year == 2025) & (LS.uses_car == 1)].employee_id
sel = pd.DataFrame({"Automobilistes 2024": profile(cu24), "Automobilistes 2025": profile(cu25),
                    "Restent": profile(stay), "Quittent la voiture": profile(leave)}).round(2)
save(sel, "table_C_selection", index=True)
note("[C] sélection (caractéristiques 2024, Lausanne) :\n" + sel.to_string())

# Élasticité intensive intra-individuelle (automobilistes les deux années), effets fixes employé
stayers = CU[CU.employee_id.isin(CU.groupby("employee_id").size().loc[lambda s: s == 2].index)]
m_fe = fit("log_days ~ log_fee + post + C(employee_id)", stayers)
note(f"[C] élasticité intensive, automobilistes les deux années, effets fixes employé + année (Lausanne vs Sion) : "
     f"{m_fe.params['log_fee']:.3f} (se {m_fe.bse['log_fee']:.3f}), N = {int(m_fe.nobs)}")
# variante sans plafonnement de E1001 : exclusion de la ligne
m_drop = fit(f"log_days ~ log_fee + lausanne + post + {CTRL}", CU[CU.car_days_raw <= 5])
note(f"[C] spécification (c) en excluant E1001-2025 au lieu de plafonner : {m_drop.params['log_fee']:.3f}")


# --- C3 : décomposition  ln(T) = ln(S) + ln(D),  T = jours voiture par employé
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
    """élasticités (ext, int, total) ; si dS, nettes de la variation à Sion (tarif inchangé)."""
    e = [np.log(dL[2025][i] / dL[2024][i]) for i in range(3)]
    if dS is not None:
        e = [e[i] - np.log(dS[2025][i] / dS[2024][i]) for i in range(3)]
    return np.array(e) / DLOGFEE


dL, dS = decomp(LS), decomp(SI)
for y in (2024, 2025):
    assert np.isclose(dL[y][0] * dL[y][1], dL[y][2])
e_raw, e_did = elast(dL), elast(dL, dS)


# Bootstrap par grappe (employés, tirés à l'intérieur de chaque site)
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
for lab, i in [("Extensive : part d'automobilistes S", 0), ("Intensive : jours par automobiliste D", 1),
               ("Totale : jours voiture par employé T = S x D", 2)]:
    fmt = (lambda v: f"{100 * v:.1f} %") if i == 0 else (lambda v: f"{v:.2f}")
    rowsC3.append({
        "Marge": lab,
        "Lausanne 2024": fmt(dL[2024][i]), "Lausanne 2025": fmt(dL[2025][i]),
        "Sion 2024": fmt(dS[2024][i]), "Sion 2025": fmt(dS[2025][i]),
        "Élasticité Lausanne avant-après": f"{e_raw[i]:.2f} [{ci_raw[0, i]:.2f} ; {ci_raw[1, i]:.2f}]",
        "Élasticité nette de Sion": f"{e_did[i]:.2f} [{ci_did[0, i]:.2f} ; {ci_did[1, i]:.2f}]",
    })
tC3 = pd.DataFrame(rowsC3)
save(tC3, "table_C3")
note(f"[C3] avant-après : ext {e_raw[0]:.3f} + int {e_raw[1]:.3f} = {e_raw[0] + e_raw[1]:.3f} ; total {e_raw[2]:.3f}")
note(f"[C3] net de Sion : ext {e_did[0]:.3f} + int {e_did[1]:.3f} = {e_did[0] + e_did[1]:.3f} ; total {e_did[2]:.3f}")
note(f"[C3] part de la marge extensive dans le total (net de Sion) : {100 * e_did[0] / e_did[2]:.0f} %")


# =============================================================================
# Partie D : scénarios 2026 (site de Lausanne, 800 employés)
# =============================================================================
N_LAU = LS.employee_id.nunique()
S4, D4 = dL[2025][0], dL[2025][1]


def scenario(eps_ext, eps_int, fees=SCENARIOS):
    """Élasticités constantes appliquées depuis la base observée à 4 CHF : X(p) = X(4) * (p/4)^eps."""
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
lo_ext, hi_ext = ci_did[:, 0]       # sensibilité : bornes de l'IC à 95 % de l'élasticité extensive
lo_int, hi_int = ci_did[:, 1]
alt_lo = scenario(lo_ext, lo_int)   # réponse forte
alt_hi = scenario(hi_ext, hi_int)   # réponse faible

# Variante linéaire issue de la LPM (c) : S(p) = S4 + a * ln(p/4)
a_c = mB["(c) + contrôles"].params["log_fee"]
lpm = central.copy()
lpm["S"] = S4 + a_c * np.log(lpm.fee / FEE_2025)
lpm["T"] = lpm.S * lpm.D
lpm["rev"] = lpm.fee * lpm["T"] * WEEKS * N_LAU

rev_2025_obs = (LS[LS.year == 2025].car_days_per_week.sum() * WEEKS * FEE_2025)
assert np.isclose(central.rev.iloc[0], rev_2025_obs)

tD1 = pd.DataFrame({
    "Tarif (CHF/jour)": central.fee.map(lambda p: f"{p} (base 2025)" if p == 4 else f"{p}"),
    "Part voiture (%)": central.S.map(pct),
    "Jours par automobiliste": central.D.map(lambda v: f"{v:.2f}"),
    "Jours voiture par employé": central["T"].map(lambda v: f"{v:.2f}"),
    "Recette annuelle (milliers CHF)": central.rev.map(lambda v: f"{v / 1000:.0f}"),
    "Sensibilité : part voiture, IC95 (%)": [f"{pct(l)} à {pct(h)}" if p != 4 else "" for p, l, h in
                                             zip(central.fee, alt_lo.S, alt_hi.S)],
    "Sensibilité : avant-après sans Sion, part voiture (%)": alt_raw.S.map(pct),
    "Sensibilité : LPM (c) linéaire, part voiture (%)": lpm.S.map(pct),
    "Sensibilité : recette, IC95 (milliers CHF)": [f"{l / 1000:.0f} à {h / 1000:.0f}" if p != 4 else "" for p, l, h in
                                                  zip(central.fee, alt_lo.rev, alt_hi.rev)],
})
save(tD1, "table_D1")
note(f"[D] élasticités centrales (nettes de Sion) : extensive {e_did[0]:.3f}, intensive {e_did[1]:.3f}")
note("[D] scénario central :\n" + central.round(4).to_string())
note(f"[D] recette 2024 observée Lausanne : {LS[LS.year == 2024].car_days_per_week.sum() * WEEKS * FEE_2024:,.0f} CHF")

# --- Figure D1 : deux panneaux (pas de double axe)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
x = central.fee
for ax, col, scale, ylab, title in [
        (axes[0], "S", 100, "Part d'employés venant en voiture (%)", "Part voiture"),
        (axes[1], "rev", 1e-3, "Recette annuelle (milliers CHF)", "Recette de parking")]:
    ax.fill_between(x, alt_lo[col] * scale, alt_hi[col] * scale, color=BLUE, alpha=0.15, lw=0,
                    label="IC 95 % des élasticités")
    ax.plot(x, central[col] * scale, color=BLUE, lw=2, marker="o", ms=4, label="Central (net de Sion)")
    ax.plot(x, alt_raw[col] * scale, color=ORANGE, lw=2, ls="--", marker="s", ms=4,
            label="Lausanne avant-après")
    ax.axvspan(8, 10.5, color="#efefec", zorder=0)
    ax.set_xlim(3.7, 10.3)
    ax.set_xticks(SCENARIOS)
    ax.set_xlabel("Tarif journalier (CHF)")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.grid(axis="y", color="#e2e1dc", lw=0.6)
    ax.text(8.1, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()), "au-delà d'un\nnouveau doublement",
            fontsize=7, color=MUTED, va="bottom")
axes[0].legend(frameon=False, fontsize=7, loc="upper right")
fig.tight_layout()
fig.savefig(FIG / "figure_D1.pdf")
fig.savefig(FIG / "figure_D1.png", dpi=200)
plt.close(fig)


# =============================================================================
# Partie E : incidence (dépense annuelle de parking en % du revenu brut observé)
# =============================================================================
E = df[df.year == 2025].copy()
E["spend"] = E.parking_fee_chf * E.car_days_per_week * WEEKS
E["burden"] = 100 * E.spend / E.gross_income_chf
# projection au tarif recommandé (Lausanne seulement) : élasticités propres à chaque classe d'accès TP
# (Lausanne avant-après, Tables B2 et C1), car l'accès TP est la dimension où la réponse diffère le plus ;
# chaque automobiliste 2025 conduit encore avec probabilité (p/4)^eps_ext et réduit ses jours de (p/4)^eps_int
ratio = RECO_FEE / FEE_2025
eps_cls = {c: (ext_stats(LS[LS.transit_class == c])[3], int_stats(LS[LS.transit_class == c])[2]) for c in ORDER_T}
note("[E] élasticités (ext, int) par classe TP utilisées pour la projection :",
     {c: (round(a, 3), round(b, 3)) for c, (a, b) in eps_cls.items()})
e_ext_i = E.transit_class.map(lambda c: eps_cls[c][0]).astype(float)
e_int_i = E.transit_class.map(lambda c: eps_cls[c][1]).astype(float)
E["burden_reco_driver"] = np.where(E.uses_car == 1, 100 * RECO_FEE * E.car_days_per_week * ratio ** e_int_i
                                   * WEEKS / E.gross_income_chf, np.nan)
E["burden_reco_all"] = np.where(E.uses_car == 1, E.burden_reco_driver * ratio ** e_ext_i, 0.0)

EL = E[E.site == "Lausanne"]
rows = []
for t, l, g in [("Ensemble", "Tous", EL)] + [(t, lev, EL[EL[v] == lev]) for t, v, lv in GROUPS for lev in lv]:
    dr = g[g.uses_car == 1]
    rows.append({"Dimension": t, "Groupe": l, "N employés": len(g), "N automobilistes 2025": len(dr),
                 "(a) Tous les employés, 2025 (%)": f"{g.burden.mean():.2f}",
                 "(b) Automobilistes 2025 (%)": f"{dr.burden.mean():.2f}",
                 f"(b) Dépense moyenne automobilistes 2025 (CHF/an)": f"{dr.spend.mean():.0f}",
                 f"(c) Tous, à {RECO_FEE} CHF (%)": f"{g.burden_reco_all.mean():.2f}",
                 f"(c) Automobilistes restants, à {RECO_FEE} CHF (%)": f"{dr.burden_reco_driver.mean():.2f}",
                 "Part de la recette 2025 (%)": pct(g.spend.sum() / EL.spend.sum())})
ES = E[E.site == "Sion"]
drS = ES[ES.uses_car == 1]
rows.append({"Dimension": "Référence", "Groupe": "Sion (2 CHF)", "N employés": len(ES),
             "N automobilistes 2025": len(drS), "(a) Tous les employés, 2025 (%)": f"{ES.burden.mean():.2f}",
             "(b) Automobilistes 2025 (%)": f"{drS.burden.mean():.2f}",
             "(b) Dépense moyenne automobilistes 2025 (CHF/an)": f"{drS.spend.mean():.0f}",
             f"(c) Tous, à {RECO_FEE} CHF (%)": "", f"(c) Automobilistes restants, à {RECO_FEE} CHF (%)": "",
             "Part de la recette 2025 (%)": ""})
tE1 = pd.DataFrame(rows)
save(tE1, "table_E1")
note(f"[E] taux d'effort max individuel 2025 (Lausanne) : {EL.burden.max():.2f} % ; "
     f"part des automobilistes au-dessus de 1 % : {pct((EL[EL.uses_car == 1].burden > 1).mean())} %")

(OUT / "results.txt").write_text("\n".join(LOG) + "\n")
print(f"\nTableaux écrits dans {TAB}, figure dans {FIG}")
