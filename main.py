import time
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
import requests


# =========================
# Configurações
# =========================
CAMERA_INDEX = 0
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
WINDOW_NAME = "Eye Tracking - Calibracao e Tempo Real"

API_URL = "http://localhost:8000/gaze"
SEND_INTERVAL_SEC = 0.10  # frequência de envio para API
REQUEST_TIMEOUT_SEC = 1.0

CALIBRATION_SECONDS_PER_POINT = 2.0
CALIBRATION_POINTS = [
    (0.05, 0.05),  # canto superior esquerdo
    (0.95, 0.05),  # canto superior direito
    (0.95, 0.95),  # canto inferior direito
    (0.05, 0.95),  # canto inferior esquerdo
]

SMOOTHING_ALPHA = 0.25  # 0 = sem atualização, 1 = sem suavização


# =========================
# Landmarks do MediaPipe
# =========================
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374

RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145


@dataclass
class EyeFeatures:
    x_ratio: float
    y_ratio: float


def _to_px(landmark, frame_w: int, frame_h: int) -> np.ndarray:
    return np.array([landmark.x * frame_w, landmark.y * frame_h], dtype=np.float32)


def _iris_center(landmarks, iris_ids, frame_w: int, frame_h: int) -> np.ndarray:
    pts = np.array([_to_px(landmarks[i], frame_w, frame_h) for i in iris_ids], dtype=np.float32)
    return np.mean(pts, axis=0)


def _eye_ratios(
    landmarks,
    iris_ids,
    outer_id,
    inner_id,
    top_id,
    bottom_id,
    frame_w: int,
    frame_h: int,
) -> tuple[float, float] | None:
    iris = _iris_center(landmarks, iris_ids, frame_w, frame_h)
    p_outer = _to_px(landmarks[outer_id], frame_w, frame_h)
    p_inner = _to_px(landmarks[inner_id], frame_w, frame_h)
    p_top = _to_px(landmarks[top_id], frame_w, frame_h)
    p_bottom = _to_px(landmarks[bottom_id], frame_w, frame_h)

    horiz_vec = p_outer - p_inner
    vert_vec = p_bottom - p_top

    horiz_den = np.dot(horiz_vec, horiz_vec)
    vert_den = np.dot(vert_vec, vert_vec)

    if horiz_den < 1e-6 or vert_den < 1e-6:
        return None

    x_ratio = float(np.dot(iris - p_inner, horiz_vec) / horiz_den)
    y_ratio = float(np.dot(iris - p_top, vert_vec) / vert_den)
    return x_ratio, y_ratio


def extract_eye_features(landmarks, frame_w: int, frame_h: int) -> EyeFeatures | None:
    left = _eye_ratios(
        landmarks,
        LEFT_IRIS,
        LEFT_EYE_OUTER,
        LEFT_EYE_INNER,
        LEFT_EYE_TOP,
        LEFT_EYE_BOTTOM,
        frame_w,
        frame_h,
    )
    right = _eye_ratios(
        landmarks,
        RIGHT_IRIS,
        RIGHT_EYE_OUTER,
        RIGHT_EYE_INNER,
        RIGHT_EYE_TOP,
        RIGHT_EYE_BOTTOM,
        frame_w,
        frame_h,
    )

    if left is None or right is None:
        return None

    return EyeFeatures(
        x_ratio=(left[0] + right[0]) / 2.0,
        y_ratio=(left[1] + right[1]) / 2.0,
    )


def fit_affine(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # features: (N, 2) ; targets: (N, 2)
    # x = a0*fx + a1*fy + a2
    # y = b0*fx + b1*fy + b2
    n = features.shape[0]
    design = np.hstack([features, np.ones((n, 1), dtype=np.float32)])
    params, _, _, _ = np.linalg.lstsq(design, targets, rcond=None)
    return params  # shape (3,2)


def map_with_affine(params: np.ndarray, feat: EyeFeatures) -> tuple[float, float]:
    p = np.array([feat.x_ratio, feat.y_ratio, 1.0], dtype=np.float32)
    out = p @ params
    return float(out[0]), float(out[1])


def post_gaze(api_url: str, x: float, y: float, frame_w: int, frame_h: int) -> None:
    payload = {
        "timestamp": time.time(),
        "x": x,
        "y": y,
        "x_norm": x / frame_w,
        "y_norm": y / frame_h,
        "frame_width": frame_w,
        "frame_height": frame_h,
    }
    try:
        requests.post(api_url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
    except requests.RequestException:
        # Falha de rede/API não deve interromper o tracking
        pass


def draw_target(frame: np.ndarray, x: int, y: int, text: str) -> None:
    cv2.circle(frame, (x, y), 16, (0, 0, 255), -1)
    cv2.circle(frame, (x, y), 30, (255, 255, 255), 2)
    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def run() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    calibration_features: list[list[float]] = []
    calibration_targets: list[list[float]] = []

    calib_idx = 0
    calib_started_at = time.time()
    affine_params = None

    smooth_x, smooth_y = None, None
    last_send = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        frame_h, frame_w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        feature = None
        if result.multi_face_landmarks:
            feature = extract_eye_features(result.multi_face_landmarks[0].landmark, frame_w, frame_h)

        # Etapa 1: calibração
        if calib_idx < len(CALIBRATION_POINTS):
            tx_norm, ty_norm = CALIBRATION_POINTS[calib_idx]
            tx = int(tx_norm * frame_w)
            ty = int(ty_norm * frame_h)

            elapsed = time.time() - calib_started_at
            countdown = max(0.0, CALIBRATION_SECONDS_PER_POINT - elapsed)
            draw_target(frame, tx, ty, f"Calibracao {calib_idx+1}/{len(CALIBRATION_POINTS)} - olhe para o ponto ({countdown:.1f}s)")

            if feature is not None:
                calibration_features.append([feature.x_ratio, feature.y_ratio])
                calibration_targets.append([tx, ty])

            if elapsed >= CALIBRATION_SECONDS_PER_POINT:
                calib_idx += 1
                calib_started_at = time.time()

                if calib_idx == len(CALIBRATION_POINTS):
                    feats = np.array(calibration_features, dtype=np.float32)
                    tars = np.array(calibration_targets, dtype=np.float32)
                    if len(feats) >= 4:
                        affine_params = fit_affine(feats, tars)
                    else:
                        raise RuntimeError("Calibração falhou: amostras insuficientes.")

        # Etapa 2: tracking em tempo real + envio API
        else:
            cv2.putText(
                frame,
                "Tracking ativo (ESC para sair)",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if feature is not None and affine_params is not None:
                gx, gy = map_with_affine(affine_params, feature)
                gx = float(np.clip(gx, 0, frame_w - 1))
                gy = float(np.clip(gy, 0, frame_h - 1))

                if smooth_x is None:
                    smooth_x, smooth_y = gx, gy
                else:
                    smooth_x = smooth_x + SMOOTHING_ALPHA * (gx - smooth_x)
                    smooth_y = smooth_y + SMOOTHING_ALPHA * (gy - smooth_y)

                cv2.circle(frame, (int(smooth_x), int(smooth_y)), 12, (0, 255, 255), -1)
                cv2.putText(
                    frame,
                    f"Gaze: ({int(smooth_x)}, {int(smooth_y)})",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                now = time.time()
                if now - last_send >= SEND_INTERVAL_SEC:
                    post_gaze(API_URL, smooth_x, smooth_y, frame_w, frame_h)
                    last_send = now

        cv2.imshow(WINDOW_NAME, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
