"""Collection of the raw public transport accessibility indicators (Part A).

This script is NOT needed to reproduce the results: analysis.py reads transit_access.csv.
It documents how the indicators were obtained.

Sources
- PT timetables: transport.opendata.ch API (official SBB / opentransportdata.swiss timetable data, including
  PostBus and tl buses). Reference day: Tuesday 10.11.2026 (2026 timetable).
- Car times: Google Maps, recorded manually on 03.11.2026 for Tuesday 10.11.2026, arriving at 08:00, from the
  origin stop to the destination station (file tools/car_times_google.csv). Google gives a range
  "typically X to Y min" based on usual traffic: we use the midpoint of the range.
- Road distance (validation only): OSRM (router.project-osrm.org, OpenStreetMap data).

Why Google Maps for car times: it accounts for usual peak-hour traffic, which matters for a commute arriving at
08:00. A free-flow time (OSRM) underestimates car time by about half and penalises short urban trips in particular.
Rejected alternative: congested car times from ARE (national passenger transport model 2023, Zenodo, CC BY 4.0),
because they are daily averages rather than peak hour, and are given between traffic zones rather than
municipalities.
Limitations: manual recording, so it cannot be re-run automatically (tools/car_times_google.csv keeps a trace, with
the date and one link per trip); ranges are sometimes wide (Bottens 24 to 50 min); Google times depend on the date
of the query.

Score: weighted mean of the z-scores of -log(PT/car ratio) (weight 0.4), log(departures 06:00-09:00) (0.4) and
-first arrival time (0.2), rescaled from 0 (least well served municipality) to 100 (best served).
Weights: a realistic trip arriving around 08:00 and the choice of morning departures count equally; the first
possible arrival counts less (usually before 06:00, so it rarely constrains an 08:00 start).
Ratio limitation: on a very short trip, a few minutes' difference gives a high ratio (Sion, Planta: 6 min by PT
vs 4 min by car, ratio 1.5, the highest of the 39 municipalities), which understates PT access in town centres.
Classes: score terciles computed separately for each site (Poor < Medium < Good): 8 municipalities per class for
Lausanne, 5 for Sion. Reason: a single ranking across both sites put almost all Valais villages in Poor and left
only 3 Lausanne municipalities (43 employees) in Lausanne's Poor class, too few for the group tables.
Consequence: a class compares municipalities of the same site, and "Poor" does not mean exactly the same level of
service in Lausanne and in Sion.

Destination: Lausanne station (8501120) for the Lausanne site, Sion station (8501506) for the Sion site.
Reasons: the exact site addresses are not given; the station is the hub of the PT network and an official stop with
a fixed identifier; car and PT are computed between the same points. Limitation: the last leg between the
station and the site is not counted.

Origin: one stop per municipality, at its centre or its railway station (ORIGINS dictionary below).
Lausanne: all Lausanne residents in the data have postcode 1004 (Chauderon, Beaulieu, Maupas, Valency, Prélaz
neighbourhoods). The chosen stop is Lausanne, Chauderon (8592010): geographic centre of this area and its main PT
hub (m1 metro, many tl lines). The official timetable has no "Maupas" stop.
Limitations: (1) a single stop represents the whole area, while residents of Prélaz or Valency have a slightly
longer trip to the station; (2) the Chauderon-station trip is short (4 min PT, 5.5 min car), so Lausanne's PT/car
ratio is sensitive to a minute or two; (3) Lausanne employees report 3.0 km on average to the site vs 1.7 km
Chauderon-station, a sign that the site is not at the station.
Sion area: Sion (postcode 1950) starts from Sion, Planta, in the town centre (same logic as Chauderon). Nendaz
(1997, Haute-Nendaz) starts from Haute-Nendaz, station/poste. Conthey starts from Châteauneuf-Conthey railway
station, in the plain; limitation: part of the population lives on the hillside (Conthey bourg), further from the
train, so Conthey's PT access is probably overstated.

Output: tools/transit_raw.csv (raw indicators) + JSON responses in tools/raw/.
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

# Weights of the three indicators in the score (same values in analysis.py)
W_RATIO, W_DEP, W_FIRST = 0.4, 0.4, 0.2

# Origin stop chosen for each municipality (central stop or the municipality's railway station)
ORIGINS = {
    "Lausanne": {
        "Lausanne": "Lausanne, Chauderon",  # postcode 1004: Chauderon, Beaulieu, Maupas, Valency, Prélaz
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
        "Sion": "Sion, Planta",  # postcode 1950: town centre
        "Arbaz": "Arbaz, poste",
        "Ardon": "Ardon",
        "Ayent": "Botyre (Ayent), centre",
        "Bramois": "Bramois, Pont de Bramois",
        "Conthey": "Châteauneuf-Conthey",
        "Evolene": "Evolène, Village",
        "Grimisuat": "Grimisuat, anc. poste",
        "Heremence": "Hérémence",
        "Nendaz": "Haute-Nendaz, station/poste",  # postcode 1997: Haute-Nendaz
        "Saint-Martin": "St-Martin VS, poste",
        "Salins": "Salins, Turin",
        "Savieze": "Savièse, St-Germain centre",
        "Vetroz": "Vétroz, poste",
        "Vex": "Vex, poste",
    },
}


def get(url, params, cache_name):
    path = RAW / f"{cache_name.replace('/', '_')}.json"
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


def hm(ts):  # "2026-11-10T07:43:00+0100" -> minutes since midnight
    h, m = ts[11:16].split(":")
    return int(h) * 60 + int(m)


def collect(site, commune, stop_name):
    o, d = station(stop_name), station(DEST[site])
    # 1) PT time for arrival around 08:00: shortest duration among arrivals between 07:00 and 08:00
    cons = connections(o["id"], d["id"], "08:00", arrival=True)
    cand = [c for c in cons if 7 * 60 <= hm(c["to"]["arrival"]) <= 8 * 60]
    best = min(cand, key=lambda c: hm(c["to"]["arrival"]) - hm(c["from"]["departure"]))
    pt_min = hm(best["to"]["arrival"]) - hm(best["from"]["departure"])
    # 2) departures 06:00 to 09:00 (distinct departure times from the origin stop)
    #    (the API ignores pagination: we re-query after the last departure returned)
    deps, t0 = set(), 6 * 60
    for _ in range(8):
        cons = connections(o["id"], d["id"], f"{t0 // 60:02d}:{t0 % 60:02d}")
        times = [hm(c["from"]["departure"]) for c in cons]
        deps.update(t for t in times if 6 * 60 <= t < 9 * 60)
        if not times or max(times) >= 9 * 60 or max(times) + 1 <= t0:
            break
        t0 = max(times) + 1
    # 3) first possible arrival at the site
    cons = connections(o["id"], d["id"], "04:00")
    first_arr = min(hm(c["to"]["arrival"]) for c in cons)
    # 4) car time (OSRM, free flow)
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

    # Car times recorded manually on Google Maps (with traffic), if complete
    gpath = Path(__file__).parent / "car_times_google.csv"
    g = pd.read_csv(gpath) if gpath.exists() else None
    use_google = g is not None and g[["car_min_low", "car_min_high"]].notna().all().all()
    if use_google:
        g["car_time_google"] = (g.car_min_low + g.car_min_high) / 2   # midpoint of the "typically X to Y min" range
        raw = raw.merge(g[["site", "home_commune", "car_min_low", "car_min_high", "car_time_google", "car_km", "query_date"]],
                        on=["site", "home_commune"], how="left", validate="one_to_one")
        raw["car_time_min"] = raw.car_time_google.round(1)
        raw["car_km_osrm"] = raw.car_km.fillna(raw.car_km_osrm)   # Google distance if recorded, otherwise OSRM
        raw = raw.drop(columns=["car_time_google", "car_km"])
    else:
        print("WARNING: tools/car_times_google.csv incomplete, OSRM car times (no traffic) used.")
    raw.to_csv(Path(__file__).parent / "transit_raw.csv", index=False)

    # Composite index (same rule as in analysis.py, which recomputes and checks it)
    t = raw.copy()
    t["pt_car_ratio"] = (t.pt_time_min / t.car_time_min).round(2)
    first = t.first_arrival.str[:2].astype(int) * 60 + t.first_arrival.str[3:].astype(int)
    z = lambda x: (x - x.mean()) / x.std()
    s = W_RATIO * z(-np.log(t.pt_car_ratio)) + W_DEP * z(np.log(t.departures_06_09)) + W_FIRST * z(-first)
    t["transit_score"] = (100 * (s - s.min()) / (s.max() - s.min())).round(1)   # 0 = worst municipality, 100 = best
    # classes: score terciles computed separately for each site
    t["transit_class"] = s.groupby(t.site).transform(lambda x: pd.qcut(x, 3, labels=["Poor", "Medium", "Good"]).astype(str))
    t["source_pt"] = ("transport.opendata.ch/v1/connections (official SBB/opentransportdata.swiss timetable), "
                      "Tuesday " + DATE + ", stop " + t.origin_stop_id.astype(str) + " -> " + t.dest_stop_id.astype(str))
    if use_google:
        t["source_car"] = ("Google Maps, recorded manually on " + t.query_date.astype(str) + " for Tuesday " + DATE + ", car, arrive by 08:00, "
                           "midpoint of the 'typically X to Y min' range (usual traffic), origin stop -> station")
    else:
        t["source_car"] = "OSRM router.project-osrm.org (OpenStreetMap), free-flow time, origin stop -> destination station"
    t["coding_notes"] = (
        "pt_time_min: shortest PT travel time among arrivals between 07:00 and 08:00; "
        "departures_06_09: number of distinct departure times from the origin stop between 06:00 and 08:59 "
        "with a connection to the destination; first_arrival: first possible arrival of the day; "
        "car_time_min: Google Maps car time (midpoint of the range, usual traffic, arrive by 08:00); "
        "car_min_low, car_min_high: bounds of the 'typically X to Y min' range recorded on Google Maps "
        "(equal when Google shows a single time); "
        "car_km_osrm: OSRM road distance, used for validation only; "
        "transit_score: weighted mean of the z-scores of -log(PT/car ratio) (weight 0.4), log(departures) (0.4) "
        "and -first arrival time (0.2) (computed over the 39 municipalities), rescaled from 0 (least well served) to 100 "
        "(best served); transit_class: score terciles computed separately for each site (Poor < Medium < Good)."
    )
    t.loc[(t.site == "Sion") & (t.home_commune == "Conthey"), "coding_notes"] += (
        " Conthey: origin at Châteauneuf-Conthey railway station (plain); hillside residents live further from the "
        "train, so PT access is probably overstated.")
    t.loc[(t.site == "Sion") & (t.home_commune == "Sion"), "coding_notes"] += (
        " Sion: postcode 1950, origin at the Planta stop (town centre).")
    t.loc[(t.site == "Sion") & (t.home_commune == "Nendaz"), "coding_notes"] += (
        " Nendaz: postcode 1997 (Haute-Nendaz), origin at Haute-Nendaz, station/poste.")
    t.loc[(t.site == "Lausanne") & (t.home_commune == "Lausanne"), "coding_notes"] += (
        " Lausanne: all employees have postcode 1004 (Chauderon, Beaulieu, Maupas, Valency, Prélaz); "
        "origin at the Chauderon stop, centre and main PT hub of the area. Short trip, so the ratio is sensitive "
        "to a minute or two.")
    cols = ["site", "home_commune", "origin_stop", "dest_stop", "pt_time_min", "pt_transfers", "car_time_min",
            "car_min_low", "car_min_high", "car_km_osrm", "pt_car_ratio", "departures_06_09", "first_arrival", "transit_score", "transit_class",
            "source_pt", "source_car", "coding_notes"]
    t[cols].sort_values(["site", "transit_score"], ascending=[True, False]).to_csv(
        Path(__file__).parent.parent / "transit_access.csv", index=False)
