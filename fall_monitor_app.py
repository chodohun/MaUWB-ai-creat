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
from fastapi import FastAPI
from scipy import signal

app = FastAPI()

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
