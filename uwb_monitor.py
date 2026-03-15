"""
UWB 위치 기반 행동 예측 모니터 (FastAPI 포함)
호흡/낙상 감지 제외 -> 위치 트래킹 및 행동 패턴 분류로 전환
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
# 1) 데이터 / 상태 정의
# ==========================

class BehaviorState(Enum):
    UNKNOWN = auto()
    STATIC = auto()       # 제자리에 정지 (가만히 있음)
    WALKING = auto()      # 일반적인 걷기
    RUNNING = auto()      # 빠른 이동 (뛰기)
    WANDERING = auto()    # 제한된 구역 내에서 방향을 자주 바꾸며 배회

@dataclass
class UWBFrame:
    """
    아두이노(UWB 3000)에서 들어오는 위치 프레임 데이터.
    엔커-태그 기반 측위 결과를 담습니다.
    """
    t: float
    tag_id: str
    x: float          # X 좌표 (미터 단위)
    y: float          # Y 좌표 (미터 단위)
    range_m: float = 0.0 # 특정 엔커와의 직접 거리 (선택 사항)


# ==========================
# 2) 행동 예측을 위한 특징(Feature) 추출
# ==========================

def extract_behavior_features(frames: list[UWBFrame]) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    일정 시간(Window) 동안 수집된 위치 데이터를 바탕으로 행동 패턴을 분석합니다.
    """
    if len(frames) < 2:
        return np.zeros(5, dtype=np.float64), {"mean_speed": 0.0, "var_x": 0.0, "var_y": 0.0}

    times = np.array([f.t for f in frames])
    xs = np.array([f.x for f in frames])
    ys = np.array([f.y for f in frames])

    # 시간 차이와 위치 차이 계산
    dt = np.diff(times)
    # 0 나누기 방지
    dt = np.clip(dt, 1e-3, None) 
    
    dx = np.diff(xs)
    dy = np.diff(ys)
    
    # 순간 이동 거리 및 속도 계산
    distances = np.sqrt(dx**2 + dy**2)
    speeds = distances / dt

    mean_speed = float(np.mean(speeds))
    max_speed = float(np.max(speeds))
    
    # 가속도 (속도의 변화량)
    accelerations = np.diff(speeds) / dt[1:] if len(speeds) > 1 else np.array([0.0])
    mean_accel = float(np.mean(np.abs(accelerations)))

    # 공간적 퍼짐 정도 (배회 패턴 등 파악용)
    var_x = float(np.var(xs))
    var_y = float(np.var(ys))
    bounding_box_area = float((np.max(xs) - np.min(xs)) * (np.max(ys) - np.min(ys)))

    # 머신러닝 모델에 입력할 최종 Feature 벡터 생성
    feat = np.array([
        mean_speed,
        max_speed,
        mean_accel,
        var_x + var_y,      # 전체적인 움직임 분산
        bounding_box_area   # 움직인 영역의 넓이
    ], dtype=np.float64)

    # NaN, Inf 방지
    feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0)

    debug = {
        "mean_speed": mean_speed,
        "max_speed": max_speed,
        "area": bounding_box_area
    }
    return feat, debug


# ==========================
# 3) AI 기반 행동 예측 모니터
# ==========================

class BehaviorPredictor:
    """
    주기적으로 UWB 위치 데이터를 모아 현재 행동을 예측합니다.
    """
    def __init__(
        self,
        window_sec: float = 3.0,
        step_sec: float = 1.0,
        model_path: str = "behavior_clf.pkl"
    ):
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.buf = deque()
        self.state = BehaviorState.UNKNOWN
        self.last_predict_t = 0.0

        # 머신러닝 모델 로드 시도 (없으면 임계값 기반 룰(Rule)엔진 사용)
        try:
            self.clf = joblib.load(model_path)
            self.use_ml = True
            print(f"[Info] ML 모델({model_path}) 로딩 성공. AI 기반 예측을 시작합니다.")
        except FileNotFoundError:
            self.clf = None
            self.use_ml = False
            print("[Warning] ML 모델이 없습니다. 임계값 기반 기본 로직으로 행동을 분류합니다.")

    def update(self, frame: UWBFrame) -> Tuple[BehaviorState, Dict[str, float]]:
        self.buf.append(frame)
        
        # 윈도우 밖의 오래된 데이터 제거
        cutoff_t = frame.t - self.window_sec
        while self.buf and self.buf[0].t < cutoff_t:
            self.buf.popleft()

        # 분석 주기(step_sec)가 지나지 않았으면 이전 상태 유지
        if frame.t - self.last_predict_t < self.step_sec:
            return self.state, {"note": "buffering"}

        self.last_predict_t = frame.t
        frames = list(self.buf)
        feat, dbg = extract_behavior_features(frames)

        # ML 모델이 있는 경우 예측
        if self.use_ml:
            try:
                # 모델이 0, 1, 2, 3으로 상태를 반환한다고 가정
                pred = int(self.clf.predict(feat.reshape(1, -1))[0])
                states_map = {0: BehaviorState.STATIC, 1: BehaviorState.WALKING, 
                              2: BehaviorState.RUNNING, 3: BehaviorState.WANDERING}
                self.state = states_map.get(pred, BehaviorState.UNKNOWN)
            except Exception as e:
                print(f"Prediction error: {e}")
        else:
            # 임계값 기반 휴리스틱 분류 (모델 학습 전 임시 사용)
            speed = dbg["mean_speed"]
            area = dbg["area"]
            
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
# 4) 사용 예시 (스트리밍 시뮬레이션)
# ==========================

def read_uwb_frame_mock(tag_id: str = "TAG_01") -> UWBFrame:
    """
    TODO: 이 함수를 아두이노(DW3000) 시리얼 포트 읽기로 교체하세요.
    """
    t = time.time()
    # 임의의 X, Y 좌표 생성 (걷는 듯한 움직임 시뮬레이션)
    x = np.sin(t * 0.5) + np.random.normal(0, 0.05)
    y = np.cos(t * 0.5) + np.random.normal(0, 0.05)
    return UWBFrame(t=t, tag_id=tag_id, x=x, y=y)

def run_demo_monitor(monitor: BehaviorPredictor, sleep_s: float = 0.1):
    print("행동 모니터링 데모를 시작합니다... (Ctrl+C로 종료)")
    while True:
        fr = read_uwb_frame_mock()
        st, info = monitor.update(fr)

        if "mean_speed" in info:
            print(f"[{time.strftime('%H:%M:%S')}] 상태: {st.name:10s} | "
                  f"속도: {info['mean_speed']:.2f}m/s | 영역: {info['area']:.2f}m^2 | "
                  f"위치: ({fr.x:.2f}, {fr.y:.2f})")
        
        time.sleep(sleep_s)


# ==========================
# 5) UWB 시리얼 파서 (아두이노 연동용)
# ==========================

def parse_arduino_serial(raw: str) -> Optional[UWBFrame]:
    """
    아두이노 시리얼 출력 예시: "TAG:T1,X:1.25,Y:3.45"
    사용 중인 아두이노 출력 포맷에 맞게 수정하세요.
    """
    try:
        parts = {p.split(':')[0].strip(): p.split(':')[1].strip() for p in raw.split(',')}
        tag_id = parts.get("TAG", "UNKNOWN")
        x = float(parts.get("X", 0.0))
        y = float(parts.get("Y", 0.0))
        return UWBFrame(t=time.time(), tag_id=tag_id, x=x, y=y)
    except Exception:
        return None

# FastAPI 및 DB 관련 설정 (이전 코드의 구조를 유지하며 단순화)
DB_PATH = "events.db"

# ==========================
# 6. 실행 엔트리포인트
# ==========================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="UWB 위치 및 행동 예측 모니터")
    parser.add_argument("--mode", choices=["monitor", "api"], default="monitor", help="실행 모드 선택")
    parser.add_argument("--model-path", type=str, default="behavior_clf.pkl", help="행동 분류 ML 모델 경로")
    parser.add_argument("--api-port", type=int, default=5000, help="FastAPI 포트")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.mode == "api":
        # FastAPI 서버 실행
        uvicorn.run(app, host="0.0.0.0", port=args.api_port)
        return

    # 모니터 모드 실행
    monitor = BehaviorPredictor(model_path=args.model_path)
    run_demo_monitor(monitor)

if __name__ == "__main__":
    main()
