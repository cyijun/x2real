import json, glob, sys, os
# usage: summ.py <run dir>... ; basket floor z (env frame) ~ -0.219 (base 0.011)
for d in sys.argv[1:]:
    f = glob.glob(d + "/eval/*/evaluation_report.jsonl")
    if not f or not os.path.getsize(f[0]):
        print(d, "no report"); continue
    r = json.loads(open(f[0]).readline())
    rows = [json.loads(l) for l in open(d + "/cloth.jsonl")]
    g = [x for x in rows if x["tag"] == "grasp" and x.get("event") in ("close", "release")]
    rel1 = next((x["step"] for x in g if x["event"] == "release"), None)
    sh = [x for x in rows if x.get("event") == "shape"]
    after1 = [x for x in sh if rel1 and rel1 + 60 <= x["step"] <= rel1 + 300]
    fold = after1[-1] if after1 else None
    close2 = next((x for x in g if x["event"] == "close" and x["step"] > (rel1 or 0)), None)
    rel2 = [x for x in g if x["event"] == "release" and close2 and x["step"] > close2["step"]]
    last = sh[-1] if sh else None
    out = {"score": "%d/%d" % (r["score"], r["max_possible_score"]), "fail": r["failed_stage"],
           "stages": [p["frame_idx"] for p in r["log"].get("progress", [])], "wall": r["wall_duration_s"]}
    if fold:
        out["fold_pair_cm"] = [round(100 * v, 1) for v in fold["pair"]]
    if close2:
        out["regrasp"] = {k: close2.get(k) for k in ("step", "captured", "nearest_m", "footprint_ratio", "self_collision_off")}
    if rel2:
        out["release_zmin_above_floor_cm"] = round(100 * (rel2[0]["zmin"] + 0.219), 1)
    if last:
        out["final"] = {"pair_cm": [round(100 * v, 1) for v in last["pair"]], "crease_cm": round(100 * last["crease_len"], 1),
                        "open_cm": round(100 * last["open_len"], 1), "top_above_floor_cm": round(100 * (last["zmax"] + 0.219), 1),
                        "speed": round(last["speed_max"], 3)}
    print(os.path.basename(d), json.dumps(out, ensure_ascii=False))
