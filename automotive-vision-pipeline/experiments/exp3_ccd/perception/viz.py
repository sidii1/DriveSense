import cv2
import config as C


def draw(frame, dets, lane, row):
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d["xyxy"])
        col = (0, 0, 255) if d["cls"] in C.VRU_IDS else (0, 200, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 3 if d["in_path"] else 1)
        label = f'{C.CLASS_NAMES[d["cls"]]}#{d["id"]} ~{d["dist_m"]:.0f}m'
        if d["ttc_s"] < 10:
            label += f' TTC {d["ttc_s"]:.1f}s'
        cv2.putText(img, label, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    if lane and lane.get("geom"):
        for xb, yb, xt, yt in lane["geom"]:
            cv2.line(img, (int(xb), int(yb)), (int(xt), int(yt)), (255, 200, 0), 3)
    txt = [f't={row["t_sec"]:.1f}s  veh={row["n_vehicles"]}  VRU={row["n_vru"]}',
           f'max 1/TTC(path)={row["max_inv_ttc_path"]:.2f}  lane_offset={row["lane_offset"]:.2f}',
           f'bright={row["brightness"]:.2f}  glare={row["glare_frac"]:.3f}']
    for i, t in enumerate(txt):
        cv2.putText(img, t, (10, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return img
