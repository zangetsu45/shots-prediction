import os
import json
import math
import pandas as pd
import numpy as np
from tqdm import tqdm

# ─────────────── CONFIG ───────────────
BASE_DIR = r"../"  # <-- update to your path
MATCHES_DIR = os.path.join(BASE_DIR, "data/matches")
EVENTS_DIR = os.path.join(BASE_DIR, "data/events")
FRAMES_DIR = os.path.join(BASE_DIR, "data/three-sixty")
# ────────────────────────────────────────

def calculate_angle(x, y):
    goal_width = 7.32
    dx = 120 - x
    dy = abs(40 - y)
    if dx == 0 and dy == 0:
        return 0.0
    return abs(math.atan2(goal_width * dx, dx*dx + dy*dy - (goal_width/2)**2))

# 1) Build a map of match_id → metadata (home/away team names)
match_map = {}
for root, _, files in os.walk(MATCHES_DIR):
    for fname in files:
        if not fname.endswith(".json"):
            continue
        path = os.path.join(root, fname)
        with open(path, encoding="utf-8") as f:
            matches = json.load(f)
        for m in matches:
            mid = m["match_id"]
            h = m["home_team"]
            a = m["away_team"]
            home_name = h.get("name") if isinstance(h, dict) else h
            away_name = a.get("name") if isinstance(a, dict) else a
            match_map[mid] = {
                "home_team": home_name,
                "away_team": away_name,
                "home_score": int(m.get("home_score", 0)),
                "away_score": int(m.get("away_score", 0))
            }

all_rows = []

# 2) Process each match file under EVENTS_DIR
for fname in tqdm(os.listdir(EVENTS_DIR), desc="Matches"):
    if not fname.endswith(".json"):
        continue

    match_id = int(os.path.splitext(fname)[0])
    md = match_map.get(match_id, {})
    events_path = os.path.join(EVENTS_DIR, fname)
    frames_path = os.path.join(FRAMES_DIR, fname)

    with open(events_path, encoding="utf-8") as f:
        events = json.load(f)

    # Load corresponding 360 data
    frame_map = {}
    if os.path.exists(frames_path) and os.path.getsize(frames_path) > 0:
        try:
            with open(frames_path, encoding="utf-8") as f:
                frames = json.load(f)
            frame_map = {frame["event_uuid"]: frame for frame in frames}
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {frames_path}: {e}")

    # Precompute score at each event index
    home_goals = away_goals = 0
    score_by_index = {}
    for ev in sorted(events, key=lambda e: e["index"]):
        if ev["type"]["name"] == "Shot" and ev.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            if ev["team"]["name"] == md.get("home_team"):
                home_goals += 1
            else:
                away_goals += 1
        score_by_index[ev["index"]] = f"{home_goals}-{away_goals}"

    # Map event IDs to pass events for assist look-ups
    pass_map = {ev["id"]: ev for ev in events if ev["type"]["name"] == "Pass"}

    # 3) Extract shot events
    for ev in events:
        if ev["type"]["name"] != "Shot":
            continue

        x, y = ev.get("location", [np.nan, np.nan])
        shot = ev.get("shot", {})
        team = ev["team"]["name"]
        is_home = 1 if team == md.get("home_team") else 0

        # assist type
        assist_type = np.nan
        kp = shot.get("key_pass_id")
        if kp and kp in pass_map:
            assist_type = pass_map[kp].get("pass", {}).get("type", {}).get("name")

        # freeze_frame defenders & GK distance
        ff = shot.get("freeze_frame", [])
        def_dists = []
        for p in ff:
            if not p.get("teammate", True):
                dx = p["location"][0] - x
                dy = p["location"][1] - y
                def_dists.append(math.hypot(dx, dy))
        min_def_dist = min(def_dists) if def_dists else np.nan
        def_density_2m = sum(1 for d in def_dists if d < 2)

        # Attempt to get GK location from 360 data
        gk_dist = np.nan
        frame = frame_map.get(ev.get("id"), {})
        freeze_frame_360 = frame.get("freeze_frame", [])
        gk = next((p for p in freeze_frame_360 if p.get("keeper") is True), None)
        if gk:
            gx, gy = gk["location"]
            gk_dist = math.hypot(gx - x, gy - y)

        # Additional features
        shot_zone_basic = shot.get("zone", {}).get("name")
        big_chance = shot.get("big_chance", False)
        situation = shot.get("situation", {}).get("name")
        minute_bucket = ev.get("minute", 0) // 15

        row = {
            "match_id": match_id,
            "home_away": is_home,
            "player": ev.get("player", {}).get("name", np.nan),
            "team": team,
            "minute": ev.get("minute"),
            "second": ev.get("second"),
            "minute_bucket": minute_bucket,
            "scoreline": score_by_index.get(ev["index"], f"{md.get('home_score')}-{md.get('away_score')}"),
            "location_x": x,
            "location_y": y,
            "shot_distance": math.hypot(120 - x, 40 - y),
            "shot_angle": calculate_angle(x, y),
            "body_part": shot.get("body_part", {}).get("name"),
            "shot_type": shot.get("type", {}).get("name"),
            "technique": shot.get("technique", {}).get("name"),
            "big_chance": big_chance,
            "situation": situation,
            "shot_outcome": shot.get("outcome", {}).get("name"),
            "play_pattern": ev.get("play_pattern", {}).get("name"),
            "under_pressure": ev.get("under_pressure", False),
            "num_defenders": len(def_dists),
            "min_defender_dist": min_def_dist,
            "defender_density_2m": def_density_2m,
        }

        all_rows.append(row)

# 4) Build DataFrame & save
cols_order = [
    "match_id","home_away","player","team","minute","second","minute_bucket",
    "scoreline","location_x","location_y","shot_distance","shot_angle",
    "body_part","shot_type","technique","big_chance", "situation",
    "shot_outcome","play_pattern","under_pressure",
    "num_defenders","min_defender_dist","defender_density_2m"
]
df = pd.DataFrame(all_rows, columns=cols_order)
df.to_csv("all_shots_features.csv", index=False)
print("✅ Saved updated CSV to all_shots_features.csv")
