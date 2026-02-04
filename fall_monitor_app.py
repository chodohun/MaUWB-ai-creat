"""
Hybrid UWB fall/respiration monitor with optional FastAPI.
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
from scipy import signal

# ==========================
# 1) 데이터 / 상태 정의
# ==========================


class State(Enum):
    NORMAL = auto()
    WALKING = auto()
    LYING = auto()
    FALL_SUSPECT = auto()
    FALL_CONFIRMED = auto()
    UNRESPONSIVE = auto()


@dataclass
class UWBFrame:
    """
    UWB에서 매 프레임 들어오는 값(예시).
    breath_raw: '호흡을 담은 1D 값' (CIR 특정 bin amplitude, phase displacement 등)
    height/speed/on_floor: 넘어짐 힌트용 (없으면 0/False로 넣어도 됨)
    """

    t: float
    breath_raw: float
    height: float = 0.0
    speed: float = 0.0
    on_floor: bool = False
    conf: float = 1.0


# ==========================
# 2) 호흡 추정(신호처리 + 품질)
# ==========================


def bandpass(sig_in: np.ndarray, fs: float, lo: float, hi: float, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    b, a = signal.butter(order, [lo / nyq, hi / nyq], btype="bandpass")
    return signal.filtfilt(b, a, sig_in)


def estimate_respiration(breath: np.ndarray, fs: float) -> Dict[str, float]:
    """
    breath: 1D 호흡 시계열 (최근 window)
    fs: 샘플링 주파수(Hz)
    반환:
      resp_bpm: 분당 호흡수 추정
      amp: 호흡 대역 RMS
      snr: 대역 내 peak가 얼마나 튀는지(간단 척도)
      quality: 0~1 (대충 신뢰도)
    """
    if len(breath) < int(fs * 5):  # 최소 5초는 있어야
        return {"resp_bpm": np.nan, "amp": 0.0, "snr": 0.0, "quality": 0.0}

    x = breath.astype(np.float64)
    x = x - np.mean(x)
    x = signal.detrend(x)

    # 사람 호흡 대역(대략 0.1~0.6 Hz = 6~36 bpm)
    # 노인/수면 등 변동 고려해 대역은 조절 가능
    try:
        xb = bandpass(x, fs, lo=0.10, hi=0.60, order=4)
    except ValueError:
        return {"resp_bpm": np.nan, "amp": 0.0, "snr": 0.0, "quality": 0.0}

    amp = float(np.sqrt(np.mean(xb**2)) + 1e-12)

    # Welch PSD로 peak 찾기
    f, pxx = signal.welch(xb, fs=fs, nperseg=min(len(xb), int(fs * 10)))
    band = (f >= 0.10) & (f <= 0.60)
    if not np.any(band):
        return {"resp_bpm": np.nan, "amp": amp, "snr": 0.0, "quality": 0.0}

    fb = f[band]
    pb = pxx[band]
    peak_i = int(np.argmax(pb))
    peak_f = float(fb[peak_i])
    resp_bpm = 60.0 * peak_f

    # 간단 SNR: peak / median
    med = float(np.median(pb) + 1e-12)
    snr = float(pb[peak_i] / med)

    # quality: amp와 snr를 적당히 압축한 값(0~1)
    # (실데이터 모으면 튜닝 권장)
    q1 = np.tanh(amp * 10.0)  # amp가 커질수록 1에 수렴
    q2 = np.tanh((snr - 1.0) / 3.0)  # snr가 1보다 크면 증가
    quality = float(np.clip(0.5 * q1 + 0.5 * q2, 0.0, 1.0))

    return {"resp_bpm": resp_bpm, "amp": amp, "snr": snr, "quality": quality}


# ==========================
# 3) 특징(feature) 만들기
# ==========================


def extract_features(frames: list[UWBFrame], fs: float) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    최근 window(frames)로부터 ML feature 벡터 생성
    """
    breath = np.array([f.breath_raw for f in frames], dtype=np.float64)
    heights = np.array([f.height for f in frames], dtype=np.float64)
    speeds = np.array([f.speed for f in frames], dtype=np.float64)
    floors = np.array([1.0 if f.on_floor else 0.0 for f in frames], dtype=np.float64)
    confs = np.array([f.conf for f in frames], dtype=np.float64)

    resp = estimate_respiration(breath, fs)

    # 넘어짐/충격 힌트(있을 때만 의미 있음)
    # (height가 없으면 0으로 들어오니 자연히 영향 작아짐)
    h_drop = float(np.max(heights) - np.min(heights))
    speed_max = float(np.max(speeds))
    speed_mean = float(np.mean(speeds))
    floor_ratio = float(np.mean(floors))
    conf_mean = float(np.mean(confs))

    # 호흡 신호 불규칙도: bandpassed 신호의 '피크성' 대신 간단히 crest factor
    x = breath - np.mean(breath)
    if np.std(x) < 1e-9:
        crest = 0.0
    else:
        crest = float(np.max(np.abs(x)) / (np.sqrt(np.mean(x**2)) + 1e-12))

    # 최종 feature 벡터
    feat = np.array(
        [
            resp["resp_bpm"],
            resp["amp"],
            resp["snr"],
            resp["quality"],
            crest,
            h_drop,
            speed_max,
            speed_mean,
            floor_ratio,
            conf_mean,
        ],
        dtype=np.float64,
    )

    # NaN 방지(호흡 못 잡으면 resp_bpm NaN일 수 있음)
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    debug = {
        "resp_bpm": resp["resp_bpm"],
        "resp_quality": resp["quality"],
        "h_drop": h_drop,
        "floor_ratio": floor_ratio,
    }
    return feat, debug


# ==========================
# 4) FSM(규칙) + AI(모델) 하이브리드
# ==========================


class HybridMonitor:
    """
    - 규칙(FSM)로 FALL_SUSPECT/시간조건 관리
    - ML로 fall_prob, unresp_prob 계산
    - 둘을 합쳐 최종 State 산출
    """

    def __init__(
        self,
        fs: float = 20.0,
        window_sec: float = 20.0,
        step_sec: float = 1.0,
        fall_model_path: str = "fall_clf.pkl",
        unresp_model_path: str = "unresp_clf.pkl",
        fall_suspect_height_drop: float = 0.7,
        fall_confirm_still_sec: float = 30.0,
        unresp_still_sec: float = 300.0,
        still_speed_th: float = 0.02,
    ):
        self.fs = fs
        self.win_n = int(window_sec * fs)
        self.step_n = int(step_sec * fs)
        self.buf = deque(maxlen=self.win_n)

        # ML 모델(확률 출력 가능한 걸로 학습해두면 좋음)
        self.fall_clf = joblib.load(fall_model_path)  # ex) RandomForest, LogisticRegression
        self.unresp_clf = joblib.load(unresp_model_path)  # ex) RandomForest, LogisticRegression
        self.state = State.NORMAL
        self.last_movement_t: Optional[float] = None
        self.fall_suspect_t: Optional[float] = None

        # 임계값
        self.fall_suspect_height_drop = fall_suspect_height_drop
        self.fall_confirm_still_sec = fall_confirm_still_sec
        self.unresp_still_sec = unresp_still_sec
        self.still_speed_th = still_speed_th

        self._counter = 0

    def _ml_proba(self, clf, x: np.ndarray) -> float:
        # 이진 분류에서 "위험(1)" 확률을 반환한다고 가정
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(x.reshape(1, -1))[0]
            # 클래스가 [0,1] 순서라 가정(아니면 clf.classes_ 확인)
            if len(p) == 2:
                return float(p[1])
            return float(np.max(p))
        if hasattr(clf, "decision_function"):
            z = float(clf.decision_function(x.reshape(1, -1))[0])
            return float(1.0 / (1.0 + np.exp(-z)))
        return float(clf.predict(x.reshape(1, -1))[0])

    def update(self, frame: UWBFrame) -> Tuple[State, Dict[str, float]]:
        self.buf.append(frame)
        self._counter += 1

        if self.last_movement_t is None:
            self.last_movement_t = frame.t

        # 움직임 갱신
        if (frame.speed > self.still_speed_th) or (not frame.on_floor):
            self.last_movement_t = frame.t

        # 윈도우가 충분히 쌓이고 step마다 한 번만 판정
        if len(self.buf) < self.win_n or (self._counter % self.step_n != 0):
            return self.state, {"note": "warming_up"}

        frames = list(self.buf)
        feat, dbg = extract_features(frames, self.fs)

        fall_p = self._ml_proba(self.fall_clf, feat)
        unresp_p = self._ml_proba(self.unresp_clf, feat)

        # --- FSM: 넘어짐 의심 트리거(규칙) ---
        # height_drop가 크고 바닥 비율이 높으면 의심
        if dbg["h_drop"] > self.fall_suspect_height_drop and dbg["floor_ratio"] > 0.6:
            if self.state not in (State.FALL_SUSPECT, State.FALL_CONFIRMED):
                self.state = State.FALL_SUSPECT
                self.fall_suspect_t = frame.t

        # 넘어짐 확정: (규칙 시간조건) OR (AI가 강하게)
        if self.state == State.FALL_SUSPECT:
            still_for = frame.t - (self.last_movement_t or frame.t)
            if still_for > self.fall_confirm_still_sec or fall_p > 0.85:
                self.state = State.FALL_CONFIRMED

        # 무반응: 오래 안 움직임 + (호흡 품질 낮거나) AI 확률 높음
        still_for = frame.t - (self.last_movement_t or frame.t)
        if still_for > self.unresp_still_sec and (unresp_p > 0.7 or dbg["resp_quality"] < 0.2):
            self.state = State.UNRESPONSIVE

        # 일반 상태 보정(위험 상태가 아니면)
        if self.state not in (State.FALL_SUSPECT, State.FALL_CONFIRMED, State.UNRESPONSIVE):
            if frame.on_floor:
                self.state = State.LYING
            elif frame.speed > 0.2:
                self.state = State.WALKING
            else:
                self.state = State.NORMAL

        info = {
            "fall_prob": fall_p,
            "unresp_prob": unresp_p,
            **dbg,
        }
        return self.state, info


# ==========================
# 5) (선택) 학습용: 모델 학습 스켈레톤
# ==========================


def train_and_save_models(X: np.ndarray, y_fall: np.ndarray, y_unresp: np.ndarray):
    """
    X: (N, F) feature
    y_fall: (N,) 0/1
    y_unresp: (N,) 0/1
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(X, y_fall, test_size=0.2, random_state=42, stratify=y_fall)
    fall_clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
    fall_clf.fit(Xtr, ytr)
    print("[FALL]\n", classification_report(yte, fall_clf.predict(Xte)))
    joblib.dump(fall_clf, "fall_clf.pkl")

    Xtr, Xte, ytr, yte = train_test_split(X, y_unresp, test_size=0.2, random_state=42, stratify=y_unresp)
    unresp_clf = RandomForestClassifier(n_estimators=300, max_depth=10, class_weight="balanced", random_state=42)
    unresp_clf.fit(Xtr, ytr)
    print("[UNRESP]\n", classification_report(yte, unresp_clf.predict(Xte)))
    joblib.dump(unresp_clf, "unresp_clf.pkl")


# ==========================
# 6) 사용 예시(스트리밍)
# ==========================


def read_uwb_frame_somehow() -> UWBFrame:
    """
    TODO: 여기를 네 UWB 장비 값으로 교체.
    필수: breath_raw (초당 fs개 들어오게)
    옵션: height/speed/on_floor
    """
    t = time.time()
    breath_raw = np.sin(2 * np.pi * 0.25 * t) + 0.05 * np.random.randn()  # 0.25Hz ~ 15 bpm
    return UWBFrame(t=t, breath_raw=float(breath_raw), height=1.2, speed=0.0, on_floor=False, conf=1.0)


def run_demo_monitor(monitor: HybridMonitor, sleep_s: float = 0.05):
    while True:
        fr = read_uwb_frame_somehow()
        st, info = monitor.update(fr)

        if "fall_prob" in info:
            resp_bpm = info["resp_bpm"] if not np.isnan(info["resp_bpm"]) else 0.0
            print(
                f"{st.name:12s} fall={info['fall_prob']:.2f} unresp={info['unresp_prob']:.2f} "
                f"resp={resp_bpm:.1f}bpm rq={info['resp_quality']:.2f} hdrop={info['h_drop']:.2f}"
            )

        time.sleep(sleep_s)


# ==========================
# 7) 이벤트 DB/FastAPI
# ==========================


DB_PATH = "events.db"


class EventStatus(str, Enum):
    PENDING = "PENDING"  # 디바이스에서 막 들어온 상태
    NOTIFIED = "NOTIFIED"  # 보호자에게 알림 보낸 상태
    GUARDIAN_OK = "GUARDIAN_OK"  # 보호자가 "괜찮음" 응답
    GUARDIAN_119 = "GUARDIAN_119"  # 보호자가 119 요청


class NotificationStatus(str, Enum):
    PENDING = "PENDING"
    NOTIFIED = "NOTIFIED"
    ACK_OK = "ACK_OK"
    ACK_119 = "ACK_119"


# ==========================
# 2. 헬퍼 함수
# ==========================


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,          -- "FALL" or "UNRESPONSIVE"
            timestamp REAL NOT NULL,           -- 디바이스에서 보낸 센서 시각
            state TEXT NOT NULL,               -- HybridFallDetector 상태 문자열
            status TEXT NOT NULL,              -- PENDING / NOTIFIED / GUARDIAN_OK / GUARDIAN_119
            payload TEXT,                      -- 센서 raw 데이터(JSON 문자열)
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    cur.execute(
        """
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
        """
    )
    cur.execute(
        """
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
        """
    )
    conn.commit()
    conn.close()


def row_to_dict(row):
    if row is None:
        return None

    (
        id_,
        event_type,
        timestamp,
        state,
        status,
        payload,
        created_at,
        updated_at,
    ) = row

    payload_dict = {}
    if payload:
        try:
            payload_dict = json.loads(payload)
        except json.JSONDecodeError:
            payload_dict = {}

    return {
        "id": id_,
        "event_type": event_type,
        "timestamp": timestamp,
        "state": state,
        "status": status,
        "payload": payload_dict,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def contact_row_to_dict(row):
    if row is None:
        return None

    (
        id_,
        name,
        phone,
        resident_id,
        fingerprint_id,
        relationship,
        channel,
        enabled,
        created_at,
        updated_at,
    ) = row

    return {
        "id": id_,
        "name": name,
        "phone": phone,
        "resident_id": resident_id,
        "fingerprint_id": fingerprint_id,
        "relationship": relationship,
        "channel": channel,
        "enabled": bool(enabled),
        "created_at": created_at,
        "updated_at": updated_at,
    }


def notification_row_to_dict(row):
    if row is None:
        return None

    (
        id_,
        event_id,
        contact_id,
        status,
        channel,
        created_at,
        updated_at,
    ) = row

    return {
        "id": id_,
        "event_id": event_id,
        "contact_id": contact_id,
        "status": status,
        "channel": channel,
        "created_at": created_at,
        "updated_at": updated_at,
    }


def update_status(event_id: int, new_status: EventStatus):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE events
        SET status = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_status.value, datetime.utcnow().isoformat(), event_id),
    )
    conn.commit()
    conn.close()


def update_notification_status(notification_id: int, new_status: NotificationStatus):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE event_notifications
        SET status = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_status.value, datetime.utcnow().isoformat(), notification_id),
    )
    conn.commit()
    conn.close()


# ==========================
# 3. 보호자 알림 / 119 트리거 (Stub)
# ==========================


def notify_contact(event_row: dict, contact_row: dict):
    """
    실제 서비스에서는 여기서 텔레그램/카카오/푸시 등으로 알림을 보냄.
    지금은 콘솔 출력만.
    """

    print("[NOTIFY CONTACT]")
    print(f"  event_id={event_row['id']} -> contact_id={contact_row['id']}")
    print(
        f"  type={event_row['event_type']}, state={event_row['state']}, status={event_row['status']}, "
        f"channel={contact_row['channel']}"
    )
    print(
        f"  contact={{'name': '{contact_row['name']}', 'phone': '{contact_row['phone']}', "
        f"'relationship': '{contact_row['relationship']}'}}"
    )
    print(f"  payload={event_row['payload']}")


def trigger_119_call(event_row: dict):
    """
    보호자가 119 요청했을 때만 호출되는 안전장치용 함수.
    지금은 콘솔 출력만 하고, 나중에 실제 전화/문자 연동 추가.
    """
    print("[TRIGGER 119 CALL]")
    print(f"  event_id={event_row['id']}")
    print(f"  type={event_row['event_type']}, state={event_row['state']}")
    print("  실제 119 자동 신고 로직은 추후 구현(법/규제 검토 필요)")


def create_notification_rows(event_id: int) -> list[dict]:
    """Insert notification rows for every enabled contact."""

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT * FROM contacts WHERE enabled = 1 ORDER BY id")
    contacts = [contact_row_to_dict(r) for r in cur.fetchall()]

    now_str = datetime.utcnow().isoformat()
    notifications = []
    for contact in contacts:
        cur.execute(
            """
            INSERT INTO event_notifications (event_id, contact_id, status, channel, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event_id,
                contact["id"],
                NotificationStatus.NOTIFIED.value,
                contact["channel"],
                now_str,
                now_str,
            ),
        )
        notif_id = cur.lastrowid
        notifications.append(
            {
                "id": notif_id,
                "event_id": event_id,
                "contact_id": contact["id"],
                "status": NotificationStatus.NOTIFIED.value,
                "channel": contact["channel"],
                "created_at": now_str,
                "updated_at": now_str,
                "contact": contact,
            }
        )

    conn.commit()
    conn.close()
    return notifications


# ==========================
# 4. 라우트
# ==========================


def build_api_app():
    from fastapi import FastAPI, HTTPException, Request

    app = FastAPI()

    @app.get("/api/health")
    def health_check():
        """
        서버 살아 있는지 확인용 간단 엔드포인트.
        """
        return {"status": "ok"}


    @app.post("/api/events", status_code=201)
    async def create_event(request: Request):
        """
        디바이스(라즈베리파이 등)에서 위험 이벤트를 보낼 때 사용하는 엔드포인트.

        요청 JSON 예시:
        {
          "event_type": "FALL",             # "FALL" or "UNRESPONSIVE"
          "timestamp": 1719999999.123,      # 센서 기준 타임스탬프 (float)
          "state": "FALL_CONFIRMED",        # HybridFallDetector 상태 이름
          "height": 0.4,
          "speed": 0.01,
          "on_floor": true
        }
        """
        data = await request.json()

        event_type = data.get("event_type")
        ts = data.get("timestamp")
        state = data.get("state")

        if event_type not in ("FALL", "UNRESPONSIVE"):
            raise HTTPException(status_code=400, detail="invalid event_type")

        if ts is None or state is None:
            raise HTTPException(status_code=400, detail="timestamp and state are required")

        # 전체 payload를 JSON 문자열로 저장
        payload_str = json.dumps(data, ensure_ascii=False)

        now_str = datetime.utcnow().isoformat()

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO events (event_type, timestamp, state, status, payload, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                float(ts),
                state,
                EventStatus.PENDING.value,
                payload_str,
                now_str,
                now_str,
            ),
        )
        event_id = cur.lastrowid
        conn.commit()

        # 방금 저장한 이벤트를 dict로 변환
        cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        conn.close()

        event_row = row_to_dict(row)

        # 보호자(연락처)에게 알림 (현재는 콘솔 출력만)
        notifications = create_notification_rows(event_id)

        for notif in notifications:
            notify_contact(event_row, notif["contact"])

        # 상태를 NOTIFIED로 갱신
        update_status(event_id, EventStatus.NOTIFIED)

        return {"event_id": event_id, "status": EventStatus.NOTIFIED.value}


    @app.post("/api/events/{event_id}/guardian_response")
    async def guardian_response(event_id: int, request: Request):
        """
        보호자 앱/웹에서 이벤트에 대한 응답을 보낼 때 사용하는 엔드포인트.

        요청 JSON 예시:
        {
          "action": "OK"        # "OK" | "CALL_119"
        }
        """
        data = await request.json()
        action = data.get("action")
        contact_id = data.get("contact_id")

        if action not in ("OK", "CALL_119"):
            raise HTTPException(status_code=400, detail="invalid action")

        # 이벤트 조회
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        conn.close()

        if row is None:
            raise HTTPException(status_code=404, detail="event not found")

        event_row = row_to_dict(row)

        notification_row = None
        if contact_id is not None:
            conn = sqlite3.connect(DB_PATH)
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM event_notifications WHERE event_id = ? AND contact_id = ?",
                (event_id, contact_id),
            )
            notification_row = cur.fetchone()
            conn.close()

        if action == "OK":
            new_status = EventStatus.GUARDIAN_OK
            update_status(event_id, new_status)
            if notification_row:
                update_notification_status(notification_row[0], NotificationStatus.ACK_OK)
        else:  # CALL_119
            new_status = EventStatus.GUARDIAN_119
            update_status(event_id, new_status)
            if notification_row:
                update_notification_status(notification_row[0], NotificationStatus.ACK_119)
            # 보호자 동의가 있을 때만 119 트리거
            trigger_119_call(event_row)

        return {"event_id": event_id, "status": new_status.value}


    @app.get("/api/events/{event_id}/notifications")
    def list_notifications(event_id: int):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM event_notifications WHERE event_id = ? ORDER BY id", (event_id,))
        rows = cur.fetchall()
        conn.close()
        return [notification_row_to_dict(r) for r in rows]


    @app.get("/api/events")
    def list_events(limit: int = 20):
        """
        최근 이벤트 목록 조회용 (디버깅/관리자용).
        ?limit=10 형태로 개수 조절 가능.
        """
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT * FROM events ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        rows = cur.fetchall()
        conn.close()

        events = [row_to_dict(r) for r in rows]
        return events


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

        name = data.get("name")
        phone = data.get("phone")
        resident_id = data.get("resident_id")
        fingerprint_id = data.get("fingerprint_id")
        relationship = data.get("relationship")
        channel = data.get("channel", "sms")
        enabled = bool(data.get("enabled", True))

        if not name:
            raise HTTPException(status_code=400, detail="name is required")

        now_str = datetime.utcnow().isoformat()
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO contacts (name, phone, resident_id, fingerprint_id, relationship, channel, enabled, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                phone,
                resident_id,
                fingerprint_id,
                relationship,
                channel,
                1 if enabled else 0,
                now_str,
                now_str,
            ),
        )
        contact_id = cur.lastrowid
        conn.commit()

        cur.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        row = cur.fetchone()
        conn.close()
        return contact_row_to_dict(row)


    @app.patch("/api/contacts/{contact_id}")
    async def update_contact(contact_id: int, request: Request):
        data = await request.json()
        allowed_fields = {
            "name",
            "phone",
            "resident_id",
            "fingerprint_id",
            "relationship",
            "channel",
            "enabled",
        }
        updates = {k: v for k, v in data.items() if k in allowed_fields}

        if not updates:
            raise HTTPException(status_code=400, detail="no valid fields to update")

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        if cur.fetchone() is None:
            conn.close()
            raise HTTPException(status_code=404, detail="contact not found")

        set_clause = ", ".join([f"{k} = ?" for k in updates])
        values = [(1 if v else 0) if k == "enabled" else v for k, v in updates.items()]
        values.append(datetime.utcnow().isoformat())
        values.append(contact_id)

        cur.execute(
            f"UPDATE contacts SET {set_clause}, updated_at = ? WHERE id = ?",
            tuple(values),
        )
        conn.commit()

        cur.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,))
        row = cur.fetchone()
        conn.close()
        return contact_row_to_dict(row)


    @app.get("/api/events/{event_id}")
    def get_event(event_id: int):
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
        row = cur.fetchone()
        conn.close()

        if row is None:
            raise HTTPException(status_code=404, detail="event not found")

        return row_to_dict(row)

    return app


# ==========================
# 9) UWB 원본 로그 수집(시리얼 → CSV)
# ==========================


def _pick_field(raw: str, key: str) -> Optional[str]:
    pos = raw.find(key)
    if pos < 0:
        return None
    end = raw.find(",", pos)
    return raw[pos + len(key) : end if end >= 0 else len(raw)]


def parse_serial_frame(raw: str) -> tuple[Optional[str], Optional[float], Optional[int]]:
    """Parse a single UWB serial line ("ID=... ,RANGE=...,RSSI=...")."""

    tag = _pick_field(raw, "ID=")
    rng_str = _pick_field(raw, "RANGE=")
    rssi_str = _pick_field(raw, "RSSI=")

    rng = float(rng_str) if rng_str else None
    rssi = int(rssi_str) if rssi_str else None
    return tag, rng, rssi


def run_serial_logger(port: str, baud: int, csv_path: Path) -> None:
    """Stream UWB serial lines into a CSV log (utc, tag_id, range_m, rssi, raw)."""

    import serial  # imported lazily to keep core deps light when unused

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    ser = serial.Serial(port, baud, timeout=1)
    file_exists = csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["utc", "tag_id", "range_m", "rssi", "raw"])

        while True:
            raw = ser.readline().decode(errors="ignore").strip()
            if not raw:
                continue

            tag_id, rng, rssi = parse_serial_frame(raw)
            writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
                    tag_id or "",
                    rng if rng is not None else "",
                    rssi if rssi is not None else "",
                    raw,
                ]
            )
            f.flush()
            print(raw)


# ==========================
# 10) CSV tail → 간단 AI 라벨러
# ==========================


def csv_tail(path: Path) -> Iterator[str]:
    """Yield the raw column from a growing CSV log (uwb_log.csv)."""

    with path.open("r", encoding="utf-8") as f:
        f.readline()  # skip header
        while True:
            position = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.1)
                f.seek(position)
                continue

            try:
                row = next(csv.reader([line]))
                raw = row[-1] if len(row) >= 5 else line.strip()
                yield raw
            except Exception:
                yield line.strip()


def classify_range_series(ranges: list[float], times: list[float]) -> str:
    if len(ranges) < 4:
        return "unknown"

    r = np.array(ranges)
    t = np.array(times)
    var = float(np.var(r))
    dt = np.diff(t)
    dr = np.diff(r)
    speed = 0.0
    jerk = 0.0

    if len(dt) > 0:
        inst_speed = np.abs(dr / np.clip(dt, 1e-3, None))
        speed = float(np.median(inst_speed))
        jerk = float(np.var(inst_speed)) if len(inst_speed) > 1 else 0.0

    if speed < 0.02 and var < 0.002:
        return "object_static"
    if speed < 0.15 and jerk > 0.05:
        return "pet_candidate"
    if 0.15 <= speed <= 0.8 and jerk <= 0.05:
        return "human_candidate"
    return "unknown"


def run_range_classifier(
    src: Iterable[str], window_sec: float = 5.0, events_csv: Path = Path("events.csv")
) -> None:
    """Tail UWB CSV lines and append simple state-change events."""

    buffers: dict[str, deque[tuple[float, float]]] = defaultdict(deque)
    last_state: dict[str, str] = defaultdict(lambda: "unknown")

    events_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = events_csv.exists()
    with events_csv.open("a", newline="", encoding="utf-8") as fe:
        writer = csv.writer(fe)
        if not file_exists:
            writer.writerow(["utc", "event", "tag_id", "state", "details"])

        for raw in src:
            tag, rng, _ = parse_serial_frame(raw)
            if tag is None or rng is None:
                continue

            now = time.time()
            dq = buffers[tag]
            dq.append((now, rng))

            cutoff = now - window_sec
            while dq and dq[0][0] < cutoff:
                dq.popleft()

            times = [t for t, _ in dq]
            ranges = [x for _, x in dq]
            label = classify_range_series(ranges, times)

            if label != last_state[tag] and label != "unknown":
                utc = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                mean_speed = np.mean(np.diff(ranges)) if len(ranges) > 1 else 0.0
                writer.writerow([utc, "state_change", tag, label, f"spd={mean_speed:.3f}"])
                fe.flush()
                print(f"[{utc}] {tag}: {last_state[tag]} → {label}")
                last_state[tag] = label


# ==========================
# 11. 실행 엔트리포인트
# ==========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UWB 하이브리드 모니터 데모, 이벤트 API, 시리얼 로거, CSV 라벨러"
    )
    parser.add_argument(
        "--mode",
        choices=["monitor", "api", "logger", "csv_ai"],
        default="monitor",
        help="실행 모드 선택",
    )
    parser.add_argument("--fs", type=float, default=20.0, help="UWB 샘플링 주파수 (Hz)")
    parser.add_argument("--window-sec", type=float, default=20.0, help="호흡/움직임 윈도우 길이 (s)")
    parser.add_argument("--step-sec", type=float, default=1.0, help="평가 주기 (s)")
    parser.add_argument("--fall-model-path", type=str, default="fall_clf.pkl", help="넘어짐 모델 경로")
    parser.add_argument("--unresp-model-path", type=str, default="unresp_clf.pkl", help="무반응 모델 경로")
    parser.add_argument("--db-path", type=Path, default=Path(DB_PATH), help="SQLite DB 파일 경로")
    parser.add_argument("--api-port", type=int, default=5000, help="FastAPI 포트")
    parser.add_argument("--demo-sleep", type=float, default=0.05, help="데모 루프 sleep 간격")
    parser.add_argument("--fall-suspect-height-drop", type=float, default=0.7, help="넘어짐 의심 높이 차 임계값")
    parser.add_argument("--fall-confirm-still-sec", type=float, default=30.0, help="넘어짐 확정 정지 시간")
    parser.add_argument("--unresp-still-sec", type=float, default=300.0, help="무반응 판정 정지 시간")
    parser.add_argument("--still-speed-th", type=float, default=0.02, help="정지로 간주하는 속도 기준")
    parser.add_argument("--serial-port", type=str, default="COM6", help="UWB 시리얼 포트")
    parser.add_argument("--baud", type=int, default=115200, help="시리얼 보드레이트")
    parser.add_argument("--csv-path", type=Path, default=Path("uwb_log.csv"), help="시리얼 로그 CSV 경로")
    parser.add_argument("--input-csv", type=Path, help="태깅할 입력 CSV(uwb_log.csv)")
    parser.add_argument(
        "--events-csv", type=Path, default=Path("events.csv"), help="라벨 결과 저장 CSV"
    )
    parser.add_argument(
        "--range-window-sec", type=float, default=5.0, help="거리 기반 라벨 윈도우 길이"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    global DB_PATH
    DB_PATH = str(args.db_path)

    if args.mode == "api":
        init_db()
        app = build_api_app()
        import uvicorn

        uvicorn.run(app, host="0.0.0.0", port=args.api_port)
        return

    if args.mode == "logger":
        run_serial_logger(args.serial_port, args.baud, args.csv_path)
        return

    if args.mode == "csv_ai":
        if not args.input_csv:
            raise SystemExit("--input-csv is required in csv_ai mode")

        run_range_classifier(
            csv_tail(args.input_csv), window_sec=args.range_window_sec, events_csv=args.events_csv
        )
        return

    monitor = HybridMonitor(
        fs=args.fs,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        fall_model_path=args.fall_model_path,
        unresp_model_path=args.unresp_model_path,
        fall_suspect_height_drop=args.fall_suspect_height_drop,
        fall_confirm_still_sec=args.fall_confirm_still_sec,
        unresp_still_sec=args.unresp_still_sec,
        still_speed_th=args.still_speed_th,
    )
    run_demo_monitor(monitor, sleep_s=args.demo_sleep)


if __name__ == "__main__":
    main()
