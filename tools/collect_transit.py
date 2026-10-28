"""Collecte des indicateurs bruts d'accès aux transports publics (Partie A).

Ce script n'est PAS nécessaire pour reproduire les résultats : analysis.py lit
transit_access.csv. Il documente comment les indicateurs ont été obtenus.

Sources
- Horaires TP : API transport.opendata.ch (données horaires officielles CFF / opentransportdata.swiss,
  y compris bus CarPostal et tl). Jour de référence : mardi 10.11.2026 (horaire 2026 en vigueur).
- Temps voiture : moteur d'itinéraire OSRM (router.project-osrm.org, données OpenStreetMap),
  temps à vitesse libre, sans trafic.

Destination : gare de Lausanne (8501120) pour le site de Lausanne, gare de Sion (8501506) pour le site de Sion
(adresse exacte des sites non communiquée).

Sortie : tools/transit_raw.csv (indicateurs bruts) + réponses JSON dans tools/raw/.
"""
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

API = "https://transport.opendata.ch/v1"
OSRM = "https://router.project-osrm.org/route/v1/driving"
DATE = "2026-11-10"
RAW = Path(__file__).parent / "raw"
RAW.mkdir(exist_ok=True)

DEST = {"Lausanne": "Lausanne", "Sion": "Sion"}

# Arrêt d'origine retenu par commune (arrêt central ou gare de la commune)
ORIGINS = {
    "Lausanne": {
        "Lausanne": "Lausanne, Bellevaux",
        "Bottens": "Bottens, croisée",
        "Bussigny": "Bussigny",
        "Chavannes-pres-Renens": "Chavannes-R., Glycines",
        "Cheseaux-sur-Lausanne": "Cheseaux-sur-L., gare",
        "Corcelles-le-Jorat": "Corcelles-le-Jorat, village",
        "Crissier": "Crissier, Place Centrale",
        "Cugy": "Cugy VD, village",
        "Ecublens": "Ecublens VD, Pontet",
        "Epalinges": "Epalinges, Croisettes",
        "Froideville": "Froideville, village",
        "Le Mont-sur-Lausanne": "Le Mont-sur-L., Grand-Mont",
        "Lutry": "Lutry",
        "Mezieres": "Mézières VD, village",
        "Montpreveyres": "Montpreveyres, village",
        "Morges": "Morges",
        "Prilly": "Prilly, centre",
        "Pully": "Pully",
        "Renens": "Renens VD",
        "Romanel-sur-Lausanne": "Romanel-sur-Lausanne",
        "Ropraz": "Ropraz, village",
        "Savigny": "Savigny, centre",
        "Servion": "Servion, Clos de l'Auberge",
        "Vulliens": "Vulliens, village",
    },
    "Sion": {
        "Sion": "Sion, Grand Champsec",
        "Arbaz": "Arbaz, poste",
        "Ardon": "Ardon",
        "Ayent": "Botyre (Ayent), centre",
        "Bramois": "Bramois, Pont de Bramois",
        "Conthey": "Châteauneuf-Conthey",
        "Evolene": "Evolène, Village",
        "Grimisuat": "Grimisuat, anc. poste",
        "Heremence": "Hérémence",
        "Nendaz": "Basse-Nendaz, village",
        "Saint-Martin": "St-Martin VS, poste",
        "Salins": "Salins, Turin",
        "Savieze": "Savièse, St-Germain centre",
        "Vetroz": "Vétroz, poste",
        "Vex": "Vex, poste",
    },
}


def get(url, params, cache_name):
    path = RAW / f"{cache_name}.json"
    if path.exists():
        return json.loads(path.read_text())
    full = url + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(full, timeout=60) as r:
        data = json.load(r)
    path.write_text(json.dumps(data, ensure_ascii=False))
    time.sleep(0.4)
    return data


def station(name):
    res = get(f"{API}/locations", {"query": name, "type": "station"}, f"loc_{name}")["stations"]
    exact = [s for s in res if s["name"] == name]
    s = (exact or res)[0]
    return {"id": s["id"], "name": s["name"], "x": s["coordinate"]["x"], "y": s["coordinate"]["y"]}


def connections(o, d, t, arrival=False, page=0, tag=""):
    p = {"from": o, "to": d, "date": DATE, "time": t, "limit": 16, "page": page}
    if arrival:
        p["isArrivalTime"] = 1
    return get(f"{API}/connections", p, f"con_{o}_{d}_{t}_{int(arrival)}_{page}{tag}")["connections"]


def hm(ts):  # "2026-11-10T07:43:00+0100" -> minutes depuis minuit
    h, m = ts[11:16].split(":")
    return int(h) * 60 + int(m)


def collect(site, commune, stop_name):
    o, d = station(stop_name), station(DEST[site])
    # 1) temps TP pour arrivée vers 08:00 : plus courte durée parmi les arrivées 07:00 à 08:00
    cons = connections(o["id"], d["id"], "08:00", arrival=True)
    cand = [c for c in cons if 7 * 60 <= hm(c["to"]["arrival"]) <= 8 * 60]
    best = min(cand, key=lambda c: hm(c["to"]["arrival"]) - hm(c["from"]["departure"]))
    pt_min = hm(best["to"]["arrival"]) - hm(best["from"]["departure"])
    # 2) départs 06:00 à 09:00 (heures de départ distinctes depuis l'arrêt d'origine)
    #    (l'API ignore la pagination : on relance la requête après le dernier départ obtenu)
    deps, t0 = set(), 6 * 60
    for _ in range(8):
        cons = connections(o["id"], d["id"], f"{t0 // 60:02d}:{t0 % 60:02d}")
        times = [hm(c["from"]["departure"]) for c in cons]
        deps.update(t for t in times if 6 * 60 <= t < 9 * 60)
        if not times or max(times) >= 9 * 60 or max(times) + 1 <= t0:
            break
        t0 = max(times) + 1
    # 3) première arrivée possible au site
    cons = connections(o["id"], d["id"], "04:00")
    first_arr = min(hm(c["to"]["arrival"]) for c in cons)
    # 4) temps voiture (OSRM, vitesse libre)
    r = get(f"{OSRM}/{o['y']},{o['x']};{d['y']},{d['x']}", {"overview": "false"}, f"osrm_{o['id']}_{d['id']}")
    car_min = r["routes"][0]["duration"] / 60
    car_km = r["routes"][0]["distance"] / 1000
    return {
        "site": site, "home_commune": commune, "origin_stop": o["name"], "origin_stop_id": o["id"],
        "dest_stop": d["name"], "dest_stop_id": d["id"],
        "pt_time_min": pt_min, "pt_transfers": best["transfers"],
        "car_time_min": round(car_min, 1), "car_km_osrm": round(car_km, 1),
        "departures_06_09": len(deps),
        "first_arrival": f"{first_arr // 60:02d}:{first_arr % 60:02d}",
    }


if __name__ == "__main__":
    rows = []
    for site, communes in ORIGINS.items():
        for commune, stop in communes.items():
            row = collect(site, commune, stop)
            print(row)
            rows.append(row)
    raw = pd.DataFrame(rows)
    raw.to_csv(Path(__file__).parent / "transit_raw.csv", index=False)

    # Indice composite (même règle que dans analysis.py, qui le recalcule et le vérifie)
    t = raw.copy()
    t["pt_car_ratio"] = (t.pt_time_min / t.car_time_min).round(2)
    first = t.first_arrival.str[:2].astype(int) * 60 + t.first_arrival.str[3:].astype(int)
    z = lambda x: (x - x.mean()) / x.std()
    t["transit_score"] = ((z(-np.log(t.pt_car_ratio)) + z(np.log(t.departures_06_09)) + z(-first)) / 3).round(3)
    t["transit_class"] = pd.qcut(t.transit_score, 3, labels=["Faible", "Moyen", "Bon"]).astype(str)
    t["source_pt"] = ("transport.opendata.ch/v1/connections (horaire officiel CFF/opentransportdata.swiss), "
                      "mardi " + DATE + ", arrêt " + t.origin_stop_id.astype(str) + " -> " + t.dest_stop_id.astype(str))
    t["source_car"] = "OSRM router.project-osrm.org (OpenStreetMap), temps à vitesse libre, arrêt d'origine -> gare de destination"
    t["coding_notes"] = (
        "pt_time_min: plus courte durée TP parmi les arrivées entre 07:00 et 08:00; "
        "departures_06_09: nombre d'heures de départ distinctes de l'arrêt d'origine entre 06:00 et 08:59 "
        "avec une correspondance vers la destination; first_arrival: première arrivée possible du jour; "
        "transit_score: moyenne des z-scores de -log(ratio TP/voiture), log(départs) et -heure de première arrivée "
        "(calculés sur les 39 communes); transit_class: terciles du score entre communes (Faible < Moyen < Bon)."
    )
    cols = ["site", "home_commune", "origin_stop", "dest_stop", "pt_time_min", "pt_transfers", "car_time_min",
            "car_km_osrm", "pt_car_ratio", "departures_06_09", "first_arrival", "transit_score", "transit_class",
            "source_pt", "source_car", "coding_notes"]
    t[cols].sort_values(["site", "transit_score"], ascending=[True, False]).to_csv(
        Path(__file__).parent.parent / "transit_access.csv", index=False)
