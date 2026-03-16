"""
UWB 위치 기반 행동 예측 모니터 + FastAPI & SQLite DB 통합 버전
(호흡/낙상 감지 -> 위치 및 행동 패턴)
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request

app = FastAPI()

# ==========================
# 1) 데이터 / 상태 정의 (행동 예측용으로 교체)
# ==========================

class BehaviorState(Enum):
    UNKNOWN = auto()
    STATIC = auto()       # 정지
    WALKING = auto()      # 걷기
    RUNNING = auto()      # 뛰기 (위험 이벤트 후보)
    WANDERING = auto()    # 배회 (위험 이벤트 후보)

@dataclass
class UWBFrame:
    """아두이노(UWB 3000)에서 들어오는 위치 데이터"""
    t: float
    tag_id: str
    x: float
    y: float
    range_m: float = 0.0


# ==========================
# 2) 행동 특징(feature) 추출
# ==========================

def extract_behavior_features(frames: list[UWBFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    if len(frames) < 2:
        return np.zeros(5, dtype=np.float64), {"mean_speed": 0.0, "var_x": 0.0, "var_y": 0.0, "area": 0.0}

    times = np.array([f.t for f in frames])
    xs = np.array([f.x for f in frames])
    ys = np.array([f.y for f in frames])

    dt = np.clip(np.diff(times), 1e-3, None) 
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    distances = np.sqrt(dx**2 + dy**2)
    speeds = distances / dt

    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    
    accelerations = np.diff(speeds) / dt[1:] if len(speeds) > 1 else np.array([0.0])
    mean_accel = float(np.mean(np.abs(accelerations)))

    var_x = float(np.var(xs))
    var_y = float(np.var(ys))
    bounding_box_area = float((np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys)))

    feat = np.array([mean_speed, max_speed, mean_accel, var_x + var_y, bounding_box_area], dtype=np.float64)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    debug = {"mean_speed": mean_speed, "max_speed": max_speed, "area": bounding_box_area}
    return feat, debug


# ==========================
# 3) AI 기반 행동 예측 모니터
# ==========================

class BehaviorPredictor:
    def __init__(self, window_sec: float = 3.0, step_sec: float = 1.0, model_path: str = "behavior_clf.pkl"):
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.buf = deque()
        self.state = BehaviorState.UNKNOWN
        self.last_predict_t = 0.0

        try:
            self.clf = joblib.load(model_path)
            self.use_ml = True
        except FileNotFoundError:
            self.clf = None
            self.use_ml = False

    def update(self, frame: UWBFrame) -> Tuple[BehaviorState, Dict[str, float]]:
        self.buf.append(frame)
        
        cutoff_t = frame.t - self.window_sec
        while self.buf and self.buf[0].t < cutoff_t:
            self.buf.popleft()

        if frame.t - self.last_predict_t < self.step_sec:
            return self.state, {"note": "buffering"}

        self.last_predict_t = frame.t
        frames = list(self.buf)
        feat, dbg = extract_behavior_features(frames)

        if self.use_ml:
            try:
                pred = int(self.clf.predict(feat.reshape(1, -1))[0])
                states_map = {0: BehaviorState.STATIC, 1: BehaviorState.WALKING, 2: BehaviorState.RUNNING, 3: BehaviorState.WANDERING}
                self.state = states_map.get(pred, BehaviorState.UNKNOWN)
            except Exception:
                pass
        else:
            speed = dbg.get("mean_speed", 0.0)
            area = dbg.get("area", 0.0)
            if speed < 0.2:
                self.state = BehaviorState.STATIC
            elif speed > 1.5:
                self.state = BehaviorState.RUNNING
            elif area > 2.0 and speed < 1.0:
                self.state = BehaviorState.WANDERING
            else:
                self.state = BehaviorState.WALKING

        return self.state, dbg


# ==========================
# 4) DB 및 상태 정의 (원본 유지)
# ==========================

DB_PATH = "events.db"

class EventStatus(str, Enum):
    PENDING = "PENDING"
    NOTIFIED = "NOTIFIED"
    GUARDIAN_OK = "GUARDIAN_OK"
    GUARDIAN_119 = "GUARDIAN_119"

class NotificationStatus(str, Enum):
    PENDING = "PENDING"
    NOTIFIED = "NOTIFIED"
    ACK_OK = "ACK_OK"
    ACK_119 = "ACK_119"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,          -- 예: "RUNNING", "WANDERING" 등 행동 기반으로 사용
            timestamp REAL NOT NULL,
            state TEXT NOT NULL,
            status TEXT NOT NULL,
            payload TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT,
            resident_id TEXT,
            fingerprint_id TEXT,
            relationship TEXT,
            channel TEXT DEFAULT 'sms',
            enabled INTEGER DEFAULT 1,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS event_notifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            contact_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            channel TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY(event_id) REFERENCES events(id),
            FOREIGN KEY(contact_id) REFERENCES contacts(id)
        );
    """)
    conn.commit()
    conn.close()

def row_to_dict(row):
    if row is None: return None
    id_, event_type, timestamp, state, status, payload, created_at, updated_at = row
    try:
        payload_dict = json.loads(payload) if payload else {}
    except json.JSONDecodeError:
        payload_dict = {}
    return {
        "id": id_, "event_type": event_type, "timestamp": timestamp, "state": state,
        "status": status, "payload": payload_dict, "created_at": created_at, "updated_at": updated_at
    }

def contact_row_to_dict(row):
    if row is None: return None
    id_, name, phone, res_id, fp_id, rel, channel, enabled, cr_at, up_at = row
    return {
        "id": id_, "name": name, "phone": phone, "resident_id": res_id,
        "fingerprint_id": fp_id, "relationship": rel, "channel": channel,
        "enabled": bool(enabled), "created_at": cr_at, "updated_at": up_at
    }

def notification_row_to_dict(row):
    if row is None: return None
    id_, ev_id, c_id, status, ch, cr_at, up_at = row
    return {
        "id": id_, "event_id": ev_id, "contact_id": c_id, "status": status,
        "channel": ch, "created_at": cr_at, "updated_at": up_at
    }

def update_status(event_id: int, new_status: EventStatus):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE events SET status = ?, updated_at = ? WHERE id = ?",
                (new_status.value, datetime.utcnow().isoformat(), event_id))
    conn.commit()
    conn.close()

def update_notification_status(notification_id: int, new_status: NotificationStatus):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("UPDATE event_notifications SET status = ?, updated_at = ? WHERE id = ?",
                (new_status.value, datetime.utcnow().isoformat(), notification_id))
    conn.commit()
    conn.close()

# ==========================
# 5) 보호자 알림 / 119 트리거 (원본 유지)
# ==========================

def notify_contact(event_row: dict, contact_row: dict):
    print("\n[NOTIFY CONTACT]")
    print(f"  event_id={event_row['id']} -> contact_id={contact_row['id']}")
    print(f"  type={event_row['event_type']}, state={event_row['state']}, status={event_row['status']}")
    print(f"  contact={{'name': '{contact_row['name']}', 'phone': '{contact_row['phone']}'}}")

def trigger_119_call(event_row: dict):
    print("\n[TRIGGER 119 CALL]")
    print(f"  event_id={event_row['id']}, type={event_row['event_type']}")
    print("  실제 119 자동 신고 로직은 추후 구현")

def create_notification_rows(event_id: int) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM contacts WHERE enabled = 1 ORDER BY id")
    contacts = [contact_row_to_dict(r) for r in cur.fetchall()]
    now_str = datetime.utcnow().isoformat()
    notifications = []
    for contact in contacts:
        cur.execute("""
            INSERT INTO event_notifications (event_id, contact_id, status, channel, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (event_id, contact["id"], NotificationStatus.NOTIFIED.value, contact["channel"], now_str, now_str))
        notif_id = cur.lastrowid
        notifications.append({
            "id": notif_id, "event_id": event_id, "contact_id": contact["id"],
            "status": NotificationStatus.NOTIFIED.value, "channel": contact["channel"],
            "created_at": now_str, "updated_at": now_str, "contact": contact
        })
    conn.commit()
    conn.close()
    return notifications

# ==========================
# 6) FastAPI 라우트 (원본 유지)
# ==========================

@app.get("/api/health")
def health_check():
    return {"status": "ok"}

@app.post("/api/events", status_code=201)
async def create_event(request: Request):
    """디바이스에서 행동 이상(배회, 뛰기 등)을 보낼 때 사용하는 엔드포인트"""
    data = await request.json()
    event_type = data.get("event_type")
    ts = data.get("timestamp")
    state = data.get("state")

    if ts is None or state is None:
        raise HTTPException(status_code=400, detail="timestamp and state are required")

    payload_str = json.dumps(data, ensure_ascii=False)
    now_str = datetime.utcnow().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO events (event_type, timestamp, state, status, payload, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (event_type, float(ts), state, EventStatus.PENDING.value, payload_str, now_str, now_str))
    event_id = cur.lastrowid
    conn.commit()

    cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cur.fetchone()
    conn.close()

    event_row = row_to_dict(row)
    notifications = create_notification_rows(event_id)
    for notif in notifications:
        notify_contact(event_row, notif["contact"])

    update_status(event_id, EventStatus.NOTIFIED)
    return {"event_id": event_id, "status": EventStatus.NOTIFIED.value}

@app.post("/api/events/{event_id}/guardian_response")
async def guardian_response(event_id: int, request: Request):
    data = await request.json()
    action = data.get("action")
    contact_id = data.get("contact_id")

    if action not in ("OK", "CALL_119"):
        raise HTTPException(status_code=400, detail="invalid action")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
    row = cur.fetchone()
    
    if row is None:
        conn.close()
        raise HTTPException(status_code=404, detail="event not found")
    event_row = row_to_dict(row)

    notification_row = None
    if contact_id is not None:
        cur.execute("SELECT * FROM event_notifications WHERE event_id = ? AND contact_id = ?", (event_id, contact_id))
        notification_row = cur.fetchone()
    conn.close()

    if action == "OK":
        new_status = EventStatus.GUARDIAN_OK
        update_status(event_id, new_status)
        if notification_row:
            update_notification_status(notification_row[0], NotificationStatus.ACK_OK)
    else:
        new_status = EventStatus.GUARDIAN_119
        update_status(event_id, new_status)
        if notification_row:
            update_notification_status(notification_row[0], NotificationStatus.ACK_119)
        trigger_119_call(event_row)

    return {"event_id": event_id, "status": new_status.value}

@app.get("/api/events")
def list_events(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM events ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    conn.close()
    return [row_to_dict(r) for r in rows]

@app.get("/api/contacts")
def list_contacts():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM contacts ORDER BY id")
    rows = cur.fetchall()
    conn.close()
    return [contact_row_to_dict(r) for r in rows]

@app.post("/api/contacts", status_code=201)
async def create_contact(request: Request):
    data = await request.json()
    now_str = datetime.utcnow().isoformat()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO contacts (name, phone, resident_id, fingerprint_id, relationship, channel, enabled, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (data.get("name"), data.get("phone"), data.get("resident_id"), data.get("fingerprint_id"), 
          data.get("relationship"), data.get("channel", "sms"), 1 if data.get("enabled", True) else 0, now_str, now_str))
    contact_id = cur.lastrowid
    conn.commit()
    cur.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
    row = cur.fetchone()
    conn.close()
    return contact_row_to_dict(row)

# ==========================
# 7) 사용 예시 (스트리밍 및 데모)
# ==========================

def read_uwb_frame_mock(tag_id: str = "TAG_01") -> UWBFrame:
    """임시 위치 데이터 생성 (아두이노 연결 시 교체)"""
    t = time.time()
    x = np.sin(t * 0.5) + np.random.normal(0, 0.05)
    y = np.cos(t * 0.5) + np.random.normal(0, 0.05)
    return UWBFrame(t=t, tag_id=tag_id, x=x, y=y)

def run_demo_monitor(monitor: BehaviorPredictor, sleep_s: float = 0.1):
    print("행동 모니터링 데모를 시작합니다... (Ctrl+C로 종료)")
    while True:
        fr = read_uwb_frame_mock()
        st, info = monitor.update(fr)
        if "mean_speed" in info:
            print(f"[{time.strftime('%H:%M:%S')}] 상태: {st.name:10s} | 속도: {info['mean_speed']:.2f}m/s | 영역: {info['area']:.2f}m^2")
        time.sleep(sleep_s)

# ==========================
# 8) 실행 엔트리포인트
# ==========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UWB 행동 예측 모니터 및 FastAPI 서버")
    parser.add_argument("--mode", choices=["monitor", "api"], default="monitor", help="실행 모드 (monitor 또는 api)")
    parser.add_argument("--model-path", type=str, default="behavior_clf.pkl", help="행동 분류 ML 모델 경로")
    parser.add_argument("--api-port", type=int, default=5000, help="FastAPI 포트")
    parser.add_argument("--db-path", type=Path, default=Path(DB_PATH), help="SQLite DB 파일 경로")
    return parser.parse_args()

def main():
    args = parse_args()
    global DB_PATH
    DB_PATH = str(args.db_path)

    if args.mode == "api":
        init_db()
        print(f"FastAPI 웹 서버를 {args.api_port} 포트에서 시작합니다...")
        uvicorn.run(app, host="0.0.0.0", port=args.api_port)
        return

    monitor = BehaviorPredictor(model_path=args.model_path)
    run_demo_monitor(monitor)

if __name__ == "__main__":
    main()
