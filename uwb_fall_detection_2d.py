import os
import signal
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import paho.mqtt.client as mqtt
from dotenv import load_dotenv
from fastapi import FastAPI

# =========================
# Config
# =========================
load_dotenv()

MQTT_SERVER = os.getenv("MQTT_SERVER", "127.0.0.1")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "uwb/distances")

ROOM_WIDTH_M = float(os.getenv("ROOM_WIDTH_M", "6.0"))
ROOM_DEPTH_M = float(os.getenv("ROOM_DEPTH_M", "6.0"))

Y_HIGH = float(os.getenv("Y_HIGH", "1.2"))
Y_LOW = float(os.getenv("Y_LOW", "0.45"))
FLOOR_ANCHOR_IDX = int(os.getenv("FLOOR_ANCHOR_IDX", "0"))
CEILING_ANCHOR_IDX = int(os.getenv("CEILING_ANCHOR_IDX", "3"))

FALL_DISTANCE_DROP_M = float(os.getenv("FALL_DISTANCE_DROP_M", "0.35"))
FALL_HEIGHT_DROP_M = float(os.getenv("FALL_HEIGHT_DROP_M", "0.4"))
MOTIONLESS_SECONDS = float(os.getenv("MOTIONLESS_SECONDS", "2.5"))
MOTIONLESS_RADIUS_M = float(os.getenv("MOTIONLESS_RADIUS_M", "0.15"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

ANCHORS = {
    0: (0.0, 0.0),
    1: (ROOM_WIDTH_M, 0.0),
    2: (0.0, ROOM_DEPTH_M),
    3: (ROOM_WIDTH_M, ROOM_DEPTH_M),
}

app = FastAPI(title="UWB Fall Detection")
# TODO: FastAPI 서버 연동 시 별도 프로세스로 실행하도록 분리 예정


@app.get("/status")
def status():
    return {"service": "uwb_fall_detection_2d", "running": True}


class ScalarKalman:
    def __init__(self, q=0.01, r=0.1):
        self.q, self.r = q, r
        self.p, self.x = 1.0, None

    def update(self, z: float) -> float:
        if self.x is None:
            self.x = z
            return z
        self.p += self.q
        k = self.p / (self.p + self.r)
        self.x = self.x + k * (z - self.x)
        self.p *= 1 - k
        return self.x


@dataclass
class FallState:
    state: str = "NORMAL"
    last_height: Optional[float] = None
    last_distance: Optional[float] = None
    candidate_ts: Optional[float] = None
    motionless_since: Optional[float] = None


class FallDetector:
    def __init__(self):
        self.s = FallState()

    def _distance_trigger(self, prev_d: Optional[float], curr_d: float) -> bool:
        return prev_d is not None and (curr_d - prev_d) > FALL_DISTANCE_DROP_M

    def _height_trigger(self, prev_h: Optional[float], curr_h: float) -> bool:
        return prev_h is not None and (prev_h - curr_h) > FALL_HEIGHT_DROP_M and curr_h <= Y_LOW

    def _is_motionless(self, now: float, xy_speed: float) -> bool:
        if xy_speed <= MOTIONLESS_RADIUS_M:
            if self.s.motionless_since is None:
                self.s.motionless_since = now
            return now - self.s.motionless_since >= MOTIONLESS_SECONDS
        self.s.motionless_since = None
        return False

    def update(self, now: float, y: float, floor_distance: float, xy_speed: float) -> str:
        # 1) 거리 기반 트리거
        d_trigger = self._distance_trigger(self.s.last_distance, floor_distance)
        # 2) 높이 기반 트리거
        h_trigger = self._height_trigger(self.s.last_height, y)
        # 3) 정지 판정
        motionless = self._is_motionless(now, xy_speed)

        if self.s.state == "NORMAL" and (d_trigger or h_trigger):
            self.s.state = "SUSPECT"
            self.s.candidate_ts = now
        elif self.s.state == "SUSPECT":
            if motionless and y <= Y_LOW:
                self.s.state = "EMERGENCY_PENDING_AI"
            elif y >= Y_HIGH:
                self.s.state = "NORMAL"
                self.s.candidate_ts = None
        elif self.s.state == "EMERGENCY" and y >= Y_HIGH:
            self.s.state = "NORMAL"

        self.s.last_height = y
        self.s.last_distance = floor_distance
        return self.s.state


def parse_distances(payload: str, prev_valid: Dict[int, float]) -> Dict[int, float]:
    result = {}
    for part in payload.split(","):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        if not key.startswith("A"):
            continue
        try:
            idx = int(key[1:])
            dist_cm = float(value)
            dist_m = dist_cm / 100.0
            if 0.1 <= dist_m <= 30.0:
                result[idx] = dist_m
            elif idx in prev_valid:
                result[idx] = prev_valid[idx]
        except Exception:
            continue
    for idx in ANCHORS:
        if idx not in result and idx in prev_valid:
            result[idx] = prev_valid[idx]
    return result


def trilaterate_2d(d: Dict[int, float]) -> Optional[Tuple[float, float]]:
    try:
        if not all(i in d for i in (0, 1, 2)):
            return None
        x1, z1 = ANCHORS[0]
        x2, z2 = ANCHORS[1]
        x3, z3 = ANCHORS[2]
        r1, r2, r3 = d[0], d[1], d[2]
        A = np.array([[2 * (x2 - x1), 2 * (z2 - z1)], [2 * (x3 - x1), 2 * (z3 - z1)]])
        b = np.array([
            r1**2 - r2**2 - x1**2 + x2**2 - z1**2 + z2**2,
            r1**2 - r3**2 - x1**2 + x3**2 - z1**2 + z3**2,
        ])
        x, z = np.linalg.lstsq(A, b, rcond=None)[0]
        return float(np.clip(x, 0, ROOM_WIDTH_M)), float(np.clip(z, 0, ROOM_DEPTH_M))
    except Exception:
        return None


def gemini_decide(state_snapshot: dict) -> Optional[bool]:
    if not GEMINI_API_KEY:
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = f"""낙상 위험 판단. JSON only: {{\"emergency\": true/false}}\n{state_snapshot}"""
        resp = model.generate_content(prompt)
        txt = (resp.text or "").lower()
        return "true" in txt
    except Exception as e:
        print(f"[WARN] Gemini 실패 -> 휴리스틱 fallback 사용: {e}")
        return None


def main():
    raw_hist = {i: deque(maxlen=5) for i in ANCHORS}
    kf_dist = {i: ScalarKalman(q=0.005, r=0.06) for i in ANCHORS}
    pos_kf = {"x": ScalarKalman(0.008, 0.1), "z": ScalarKalman(0.008, 0.1), "y": ScalarKalman(0.01, 0.12)}

    fall_detector = FallDetector()
    prev_valid = {}
    latest = {"raw": {}, "kalman": {}, "final": {}, "x": None, "z": None, "y": None, "state": "NORMAL"}
    prev_xz = None

    running = True

    def _sig_handler(*_):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sig_handler)

    def on_connect(client, userdata, flags, rc, properties=None):
        if rc == 0:
            print(f"[INFO] MQTT connected: {MQTT_SERVER}, topic={MQTT_TOPIC}")
            client.subscribe(MQTT_TOPIC)
        else:
            print(f"[ERROR] MQTT connection failed. rc={rc}")

    def on_message(client, userdata, msg):
        nonlocal prev_xz, prev_valid
        try:
            payload = msg.payload.decode("utf-8", errors="ignore").strip()
            parsed = parse_distances(payload, prev_valid)
            if len(parsed) < 3:
                return
            prev_valid = parsed.copy()
            latest["raw"] = parsed.copy()

            for i, v in parsed.items():
                raw_hist[i].append(v)
                med = float(np.median(raw_hist[i]))
                latest["kalman"][i] = kf_dist[i].update(med)

            latest["final"] = latest["kalman"].copy()
            xz = trilaterate_2d(latest["final"])
            if not xz:
                return

            x, z = xz
            y_raw = latest["final"].get(FLOOR_ANCHOR_IDX, 0.0)
            x_f = pos_kf["x"].update(x)
            z_f = pos_kf["z"].update(z)
            y_f = pos_kf["y"].update(y_raw)
            latest["x"], latest["z"], latest["y"] = x_f, z_f, y_f

            speed = 0.0
            if prev_xz is not None:
                speed = float(np.linalg.norm(np.array([x_f, z_f]) - np.array(prev_xz)))
            prev_xz = (x_f, z_f)

            st = fall_detector.update(time.time(), y_f, latest["final"].get(FLOOR_ANCHOR_IDX, y_f), speed)

            if st == "EMERGENCY_PENDING_AI":
                snapshot = {"y": y_f, "speed": speed, "dist": latest["final"]}
                ai = gemini_decide(snapshot)
                # 4) Gemini 판정
                if ai is True:
                    st = "EMERGENCY"
                # 5) 휴리스틱 오버라이드
                elif ai is None and y_f <= Y_LOW and speed <= MOTIONLESS_RADIUS_M:
                    st = "EMERGENCY"
                else:
                    st = "NORMAL"
                fall_detector.s.state = st

            latest["state"] = st
        except Exception as e:
            print(f"[WARN] message handling error: {e}")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_SERVER, 1883, 30)
    except Exception as e:
        print(f"[ERROR] MQTT 초기 연결 실패: {e}")

    client.loop_start()

    plt.ion()
    fig, ax = plt.subplots(figsize=(7, 7))

    try:
        while running:
            ax.clear()
            ax.set_title("UWB X/Z Top View + Fall Status")
            ax.set_xlim(0, ROOM_WIDTH_M)
            ax.set_ylim(0, ROOM_DEPTH_M)
            ax.set_xlabel("X (m)")
            ax.set_ylabel("Z (m)")
            ax.grid(True, alpha=0.3)

            for i, (ax_x, ax_z) in ANCHORS.items():
                ax.scatter(ax_x, ax_z, c="blue", marker="^", s=110)
                ax.text(ax_x, ax_z, f"Anchor A{i}")

            if latest["x"] is not None and latest["z"] is not None:
                ax.scatter(latest["x"], latest["z"], c="red", s=120)
                ax.text(latest["x"], latest["z"], "Tag")

            raw_s = ", ".join([f"A{i}:{latest['raw'].get(i, np.nan):.2f}" for i in ANCHORS])
            kal_s = ", ".join([f"A{i}:{latest['kalman'].get(i, np.nan):.2f}" for i in ANCHORS])
            fin_s = ", ".join([f"A{i}:{latest['final'].get(i, np.nan):.2f}" for i in ANCHORS])
            status_txt = f"State={latest['state']} | Y={latest['y']}"
            ax.text(0.01, 1.02, f"raw(m): {raw_s}\nkalman(m): {kal_s}\nfinal(m): {fin_s}\n{status_txt}", transform=ax.transAxes, fontsize=8)

            plt.pause(0.1)
    finally:
        print("[INFO] shutting down...")
        client.loop_stop()
        client.disconnect()
        plt.close(fig)


if __name__ == "__main__":
    main()
