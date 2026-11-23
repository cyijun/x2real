"""mkprog.py <cloth.jsonl with a reset record> <template program.py> <out program.py> [basket_name]
Validation helper: copies the template, replacing NEAR/FAR/BASKET with the oracle layout of the
episode's first reset record (env frame -> robot base frame; robot root yaw assumed 0)."""
import json, re, sys
trace, tmpl, out = sys.argv[1:4]
bname = sys.argv[4] if len(sys.argv) > 4 else None
rec = next(json.loads(l) for l in open(trace) if '"reset"' in l)
rx, ry, rz = rec["robot_root"]
assert abs(rec["robot_quat"][3]) > 0.999 or abs(rec["robot_quat"][0]) > 0.999, rec["robot_quat"]
c = [[p[0] - rx, p[1] - ry] for p in rec["corners"]]
objs = rec.get("objects", {})
if bname is None:
    cands = [k for k in objs if "basket" in k.lower()]
    assert len(cands) == 1, objs.keys()
    bname = cands[0]
b = objs[bname]
s = open(tmpl).read()
s = re.sub(r"^NEAR = .*$", "NEAR = np.array([[%.4f, %.4f], [%.4f, %.4f]])" % (*c[0], *c[1]), s, flags=re.M)
s = re.sub(r"^FAR = .*$", "FAR = np.array([[%.4f, %.4f], [%.4f, %.4f]])" % (*c[2], *c[3]), s, flags=re.M)
s = re.sub(r"^BASKET = .*$", "BASKET = np.array([%.4f, %.4f])                    # %s centre (base xy), quat %s"
           % (b[0] - rx, b[1] - ry, bname, b[3:]), s, flags=re.M)
open(out, "w").write(s)
print("NEAR", c[:2], "FAR", c[2:], "BASKET", bname, [b[0] - rx, b[1] - ry, b[2] - rz], "quat", b[3:])
