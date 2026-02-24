import queue
import threading
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

# Processamento em resolução menor para ganhar FPS
PROCESS_SCALE = 0.5

API_URL = "http://localhost:8000/gaze"
SEND_INTERVAL_SEC = 0.10
REQUEST_TIMEOUT_SEC = 0.6

# Grade 3x3 melhora precisão espacial vs apenas cantos
CALIBRATION_SECONDS_PER_POINT = 1.6
CALIBRATION_POINTS = [
    (0.10, 0.10), (0.50, 0.10), (0.90, 0.10),
    (0.10, 0.50), (0.50, 0.50), (0.90, 0.50),
    (0.10, 0.90), (0.50, 0.90), (0.90, 0.90),
]
CALIBRATION_COLLECTION_START_RATIO = 0.40

SMOOTHING_ALPHA = 0.30
MIN_VALID_SAMPLES_PER_POINT = 8
MAX_QUEUE_SIZE = 8
L2_REG = 1e-3


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
    lx: float
    ly: float
    rx: float
    ry: float

    def as_vector(self) -> np.ndarray:
        return np.array([self.lx, self.ly, self.rx, self.ry], dtype=np.float32)


class AsyncGazeSender:
    def __init__(self, api_url: str):
        self.api_url = api_url
        self._q: queue.Queue[dict] = queue.Queue(maxsize=MAX_QUEUE_SIZE)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._session = requests.Session()
        self._thread.start()

    def send(self, payload: dict) -> None:
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._q.put_nowait(payload)
        except queue.Full:
            pass

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._session.close()

    def _worker(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                payload = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._session.post(self.api_url, json=payload, timeout=REQUEST_TIMEOUT_SEC)
            except requests.RequestException:
                pass


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
    if not (-0.6 <= x_ratio <= 1.6 and -0.6 <= y_ratio <= 1.6):
        return None
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
    return EyeFeatures(lx=left[0], ly=left[1], rx=right[0], ry=right[1])


def _poly_features(v4: np.ndarray) -> np.ndarray:
    lx, ly, rx, ry = [float(x) for x in v4]
    terms = [
        1.0,
        lx, ly, rx, ry,
        lx * lx, ly * ly, rx * rx, ry * ry,
        lx * ly, rx * ry,
        lx * rx, ly * ry,
        lx * ry, ly * rx,
    ]
    return np.array(terms, dtype=np.float32)


def fit_polynomial_map(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    # features: (N,4), targets: (N,2)
    design = np.vstack([_poly_features(v) for v in features])
    xtx = design.T @ design
    reg = L2_REG * np.eye(xtx.shape[0], dtype=np.float32)
    inv = np.linalg.inv(xtx + reg)
    weights = inv @ design.T @ targets
    return weights  # (P,2)


def map_with_polynomial(weights: np.ndarray, feat: EyeFeatures) -> tuple[float, float]:
    vec = _poly_features(feat.as_vector())
    out = vec @ weights
    return float(out[0]), float(out[1])


def draw_target(frame: np.ndarray, x: int, y: int, text: str) -> None:
    cv2.circle(frame, (x, y), 16, (0, 0, 255), -1)
    cv2.circle(frame, (x, y), 30, (255, 255, 255), 2)
    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)


def robust_center(samples: list[np.ndarray]) -> np.ndarray:
    arr = np.array(samples, dtype=np.float32)
    med = np.median(arr, axis=0)
    d = np.linalg.norm(arr - med, axis=1)
    mad = np.median(d) + 1e-6
    filtered = arr[d < (2.8 * mad)]
    if len(filtered) == 0:
        filtered = arr
    return np.median(filtered, axis=0)


def run() -> None:
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a câmera.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.6,
    )

    calibration_features: list[np.ndarray] = []
    calibration_targets: list[list[float]] = []
    current_point_samples: list[np.ndarray] = []

    calib_idx = 0
    calib_started_at = time.time()
    map_weights = None

    smooth_x, smooth_y = None, None
    last_send = 0.0
    sender = AsyncGazeSender(API_URL)

    frame_count = 0
    fps_clock = time.time()
    fps_value = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            proc_w = max(320, int(frame_w * PROCESS_SCALE))
            proc_h = max(180, int(frame_h * PROCESS_SCALE))
            proc = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            feature = None
            if result.multi_face_landmarks:
                feature = extract_eye_features(result.multi_face_landmarks[0].landmark, proc_w, proc_h)

            if calib_idx < len(CALIBRATION_POINTS):
                tx_norm, ty_norm = CALIBRATION_POINTS[calib_idx]
                tx = int(tx_norm * frame_w)
                ty = int(ty_norm * frame_h)

                elapsed = time.time() - calib_started_at
                countdown = max(0.0, CALIBRATION_SECONDS_PER_POINT - elapsed)
                draw_target(frame, tx, ty, f"Calibracao {calib_idx+1}/{len(CALIBRATION_POINTS)} ({countdown:.1f}s)")

                if feature is not None and elapsed >= (CALIBRATION_SECONDS_PER_POINT * CALIBRATION_COLLECTION_START_RATIO):
                    current_point_samples.append(feature.as_vector())

                if elapsed >= CALIBRATION_SECONDS_PER_POINT:
                    if len(current_point_samples) >= MIN_VALID_SAMPLES_PER_POINT:
                        point_feat = robust_center(current_point_samples)
                        calibration_features.append(point_feat)
                        calibration_targets.append([tx, ty])
                        calib_idx += 1
                    current_point_samples = []
                    calib_started_at = time.time()

                    if calib_idx == len(CALIBRATION_POINTS):
                        feats = np.array(calibration_features, dtype=np.float32)
                        tars = np.array(calibration_targets, dtype=np.float32)
                        if len(feats) >= 6:
                            map_weights = fit_polynomial_map(feats, tars)
                        else:
                            raise RuntimeError("Calibração falhou: amostras insuficientes.")

            else:
                cv2.putText(frame, "Tracking ativo (ESC para sair)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                if feature is not None and map_weights is not None:
                    gx, gy = map_with_polynomial(map_weights, feature)
                    gx = float(np.clip(gx, 0, frame_w - 1))
                    gy = float(np.clip(gy, 0, frame_h - 1))

                    if smooth_x is None:
                        smooth_x, smooth_y = gx, gy
                    else:
                        smooth_x = smooth_x + SMOOTHING_ALPHA * (gx - smooth_x)
                        smooth_y = smooth_y + SMOOTHING_ALPHA * (gy - smooth_y)

                    cv2.circle(frame, (int(smooth_x), int(smooth_y)), 12, (0, 255, 255), -1)
                    cv2.putText(frame, f"Gaze: ({int(smooth_x)}, {int(smooth_y)})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

                    now = time.time()
                    if now - last_send >= SEND_INTERVAL_SEC:
                        sender.send(
                            {
                                "timestamp": now,
                                "x": smooth_x,
                                "y": smooth_y,
                                "x_norm": smooth_x / frame_w,
                                "y_norm": smooth_y / frame_h,
                                "frame_width": frame_w,
                                "frame_height": frame_h,
                            }
                        )
                        last_send = now

            frame_count += 1
            now = time.time()
            dt = now - fps_clock
            if dt >= 0.5:
                fps_value = frame_count / dt
                frame_count = 0
                fps_clock = now
            cv2.putText(frame, f"FPS: {fps_value:.1f}", (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        sender.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
