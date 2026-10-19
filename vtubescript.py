
import asyncio
import json
import math
import sys
import time
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import websockets

try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    sd = None
    AUDIO_AVAILABLE = False


# =========================================================
# НАСТРОЙКИ
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CALIBRATION_PATH = SCRIPT_DIR / "face_calibration.json"
MODEL_PATH = SCRIPT_DIR / "face_landmarker.task"

CAM_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

WEBSOCKET_HOST = "127.0.0.1"
WEBSOCKET_PORT = 8765

# Сглаживание обычных каналов лица.
SMOOTHING = 0.55

# Сглаживание рта: больше значение = быстрее реакция.
MOUTH_SMOOTHING = 0.78

# Звуковой вход.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1

# Ниже этого уровня звук считается фоновым шумом.
AUDIO_NOISE_FLOOR = 0.012

# Уровень RMS, при котором достигается максимальное усиление.
AUDIO_FULL_LEVEL = 0.095

# Максимальная добавка от звука к открытию рта.
AUDIO_JAW_GAIN = 0.65

# Сглаживание атаки и отпускания звукового сигнала.
AUDIO_ATTACK = 0.72
AUDIO_RELEASE = 0.30

# Не добавлять открытие рта от очень слабого шума.
AUDIO_MIN_ACTIVITY = 0.06


# =========================================================
# КАЛИБРОВКА
# =========================================================

TARGETS = [
    ("neutral", "1/12 Нейтральное лицо (расслабься, смотри в камеру)"),
    ("smile", "2/12 Обычная улыбка (уголки рта вверх, глаза открыты)"),
    ("joy", "3/12 Радость (широкая улыбка + прищур глаз ^^)"),
    ("sad", "4/12 Грусть (уголки рта вниз, брови домиком)"),
    ("angry", "5/12 Злость (сдвинь брови вниз, нахмурься)"),
    ("surprised", "6/12 Удивление (брови вверх, широко открытые глаза и рот)"),
    ("closed", "7/12 Закрытые глаза (мягко закрой оба глаза)"),
    ("mouth_A", "8/12 Звук А (открой челюсть вертикально)"),
    ("mouth_E", "9/12 Звук Э / E (растяни губы в стороны)"),
    ("mouth_I", "10/12 Звук И / I (узкая щель губ)"),
    ("mouth_O", "11/12 Звук О / O (округли губы кольцом)"),
    ("mouth_U", "12/12 Звук У / U (вытяни губы трубочкой вперед)"),
]

BLENDSHAPE_NAMES = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft",
    "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel", "mouthLeft", "mouthLowerDownLeft",
    "mouthLowerDownRight", "mouthPressLeft", "mouthPressRight",
    "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight"
]


# =========================================================
# ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# =========================================================

calibration = {}
previous_face = {}
clients = set()

audio_lock = threading.Lock()
audio_rms = 0.0
audio_envelope = 0.0
audio_stream = None

# Время последнего кадра используется MediaPipe VIDEO.
last_timestamp_ms = 0


# =========================================================
# УТИЛИТЫ
# =========================================================

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, float(val)))


def extract_blendshapes(result):
    if not result.face_blendshapes:
        return {}

    return {
        c.category_name: float(c.score)
        for c in result.face_blendshapes[0]
    }


def get_neutral_blendshapes():
    neutral = calibration.get("neutral", {})
    return neutral.get("blendshapes", {})


def smooth_face(face):
    """
    Сглаживает каналы независимо.
    Рот получает более быструю реакцию, чем остальные части лица.
    """
    global previous_face

    output = {}

    mouth_names = (
        "jawOpen",
        "mouthClose",
        "mouthFunnel",
        "mouthPucker",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
    )

    for name in BLENDSHAPE_NAMES:
        curr = float(face.get(name, 0.0))
        prev = float(previous_face.get(name, curr))

        alpha = (
            MOUTH_SMOOTHING
            if name in mouth_names
            else SMOOTHING
        )

        output[name] = float(np.clip(
            prev + (curr - prev) * alpha,
            0.0,
            1.0
        ))

    previous_face = output
    return output


def head_rotation(result):
    matrices = getattr(
        result,
        "facial_transformation_matrixes",
        None
    )

    if not matrices or len(matrices) == 0:
        return {
            "yaw": 0.0,
            "pitch": 0.0,
            "roll": 0.0
        }

    matrix = np.array(matrices[0]).reshape(4, 4)
    rot = matrix[:3, :3]

    sy = math.sqrt(
        rot[0, 0] ** 2 + rot[1, 0] ** 2
    )

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rot[2, 1], rot[2, 2])
        y = math.atan2(-rot[2, 0], sy)
        z = math.atan2(rot[1, 0], rot[0, 0])
    else:
        x = math.atan2(-rot[1, 2], rot[1, 1])
        y = math.atan2(-rot[2, 0], sy)
        z = 0.0

    return {
        "yaw": float(np.clip(y * 1.5, -0.8, 0.8)),
        "pitch": float(np.clip(-x * 1.3, -0.6, 0.6)),
        "roll": float(np.clip(-z * 1.2, -0.6, 0.6))
    }


# =========================================================
# ЗВУК: ИЗМЕРЕНИЕ ГРОМКОСТИ МИКРОФОНА
# =========================================================

def audio_callback(indata, frames, time_info, status):
    """
    Вызывается аудиопотоком.
    Здесь только измеряем RMS; камера и WebSocket не блокируются.
    """
    global audio_rms

    if status:
        pass

    samples = np.asarray(indata, dtype=np.float32)

    if samples.size == 0:
        return

    rms = float(np.sqrt(np.mean(samples * samples)))

    with audio_lock:
        audio_rms = rms


def start_audio():
    """
    Запускает микрофон, если sounddevice и устройство доступны.
    При ошибке трекер продолжает работать только с камерой.
    """
    global audio_stream

    if not AUDIO_AVAILABLE:
        print(
            "Звук отключён: установите sounddevice "
            "командой pip install sounddevice"
        )
        return False

    try:
        audio_stream = sd.InputStream(
            samplerate=AUDIO_SAMPLE_RATE,
            channels=AUDIO_CHANNELS,
            dtype="float32",
            blocksize=512,
            callback=audio_callback
        )

        audio_stream.start()

        print("✓ Микрофон подключён.")
        return True

    except Exception as error:
        audio_stream = None
        print(f"Предупреждение: микрофон недоступен: {error}")
        print("Трекинг продолжится без звукового управления.")
        return False


def update_audio_envelope(dt):
    """
    Преобразует громкость в плавный сигнал 0..1.
    Используется разная скорость нарастания и спада.
    """
    global audio_envelope

    with audio_lock:
        rms = audio_rms

    normalized = clamp(
        (rms - AUDIO_NOISE_FLOOR) /
        max(0.001, AUDIO_FULL_LEVEL - AUDIO_NOISE_FLOOR)
    )

    if normalized < AUDIO_MIN_ACTIVITY:
        normalized = 0.0

    alpha = (
        AUDIO_ATTACK
        if normalized > audio_envelope
        else AUDIO_RELEASE
    )

    # dt-зависимое сглаживание для устойчивой реакции.
    time_factor = clamp(dt * 60.0, 0.0, 1.0)
    effective_alpha = alpha * time_factor

    audio_envelope += (
        normalized - audio_envelope
    ) * effective_alpha

    audio_envelope = clamp(audio_envelope)

    return audio_envelope, rms


def apply_audio_to_face(face, audio_strength):
    """
    Звук добавляет открытие челюсти поверх видеотрекинга.
    Формы губ по-прежнему определяются камерой.

    Используется max, чтобы звук не закрывал рот,
    если камера уже определила большее открытие.
    """
    camera_jaw = clamp(face.get("jawOpen", 0.0))

    audio_jaw = clamp(
        audio_strength * AUDIO_JAW_GAIN
    )

    combined_jaw = max(camera_jaw, audio_jaw)

    face["jawOpen"] = combined_jaw

    return combined_jaw, audio_jaw


# =========================================================
# НЕЗАВИСИМЫЕ КАНАЛЫ ЛИЦА
# Никакой главной эмоции или взаимоисключающих состояний.
# =========================================================

def compute_face_channels(face, audio_strength=0.0):
    neutral_bs = get_neutral_blendshapes()

    def delta(name):
        return max(
            0.0,
            face.get(name, 0.0) - neutral_bs.get(name, 0.0)
        )

    smile_l = clamp(face.get("mouthSmileLeft", 0.0))
    smile_r = clamp(face.get("mouthSmileRight", 0.0))

    frown_l = clamp(face.get("mouthFrownLeft", 0.0))
    frown_r = clamp(face.get("mouthFrownRight", 0.0))

    brow_down_l = clamp(face.get("browDownLeft", 0.0))
    brow_down_r = clamp(face.get("browDownRight", 0.0))

    brow_inner_up = clamp(face.get("browInnerUp", 0.0))

    brow_outer_l = clamp(face.get("browOuterUpLeft", 0.0))
    brow_outer_r = clamp(face.get("browOuterUpRight", 0.0))

    eye_squint_l = clamp(face.get("eyeSquintLeft", 0.0))
    eye_squint_r = clamp(face.get("eyeSquintRight", 0.0))

    eye_wide_l = clamp(face.get("eyeWideLeft", 0.0))
    eye_wide_r = clamp(face.get("eyeWideRight", 0.0))

    blink_l = clamp(face.get("eyeBlinkLeft", 0.0))
    blink_r = clamp(face.get("eyeBlinkRight", 0.0))

    jaw = clamp(face.get("jawOpen", 0.0))
    pucker = clamp(face.get("mouthPucker", 0.0))
    funnel = clamp(face.get("mouthFunnel", 0.0))

    stretch_l = clamp(face.get("mouthStretchLeft", 0.0))
    stretch_r = clamp(face.get("mouthStretchRight", 0.0))

    stretch = (stretch_l + stretch_r) * 0.5
    smile = (smile_l + smile_r) * 0.5
    frown = (frown_l + frown_r) * 0.5
    brow_down = (brow_down_l + brow_down_r) * 0.5
    squint = (eye_squint_l + eye_squint_r) * 0.5
    eye_wide = (eye_wide_l + eye_wide_r) * 0.5

    # Все сигналы существуют одновременно.
    # Они не выбирают одно доминирующее выражение.
    channels = {
        # Брови — независимо для каждой стороны.
        "browDownLeft": brow_down_l,
        "browDownRight": brow_down_r,
        "browInnerUp": brow_inner_up,
        "browOuterUpLeft": brow_outer_l,
        "browOuterUpRight": brow_outer_r,

        # Глаза и веки.
        "eyeBlinkLeft": blink_l,
        "eyeBlinkRight": blink_r,
        "eyeSquintLeft": eye_squint_l,
        "eyeSquintRight": eye_squint_r,
        "eyeWideLeft": eye_wide_l,
        "eyeWideRight": eye_wide_r,

        # Уголки рта — независимо.
        "mouthSmileLeft": smile_l,
        "mouthSmileRight": smile_r,
        "mouthFrownLeft": frown_l,
        "mouthFrownRight": frown_r,

        # Рот и артикуляция.
        "jawOpen": jaw,
        "mouthPucker": pucker,
        "mouthFunnel": funnel,
        "mouthStretchLeft": stretch_l,
        "mouthStretchRight": stretch_r,

        # Сводные измерения для диагностики.
        "smile": smile,
        "frown": frown,
        "squint": squint,
        "browDown": brow_down,
        "browUp": brow_inner_up,
        "eyeWide": eye_wide,
        "audioStrength": clamp(audio_strength),

        # Сигналы формы рта. Они не являются общей эмоцией.
        "phoneme_a": clamp(jaw * 1.8),
        "phoneme_e": clamp(stretch * 1.6) if jaw < 0.4 else 0.0,
        "phoneme_i": clamp(stretch * 1.8 - jaw * 0.5),
        "phoneme_o": clamp(funnel * 2.0),
        "phoneme_u": clamp(pucker * 2.0),
    }

    # Дельты относительно нейтрали доступны для диагностики.
    channels["smileDelta"] = clamp(
        (delta("mouthSmileLeft") + delta("mouthSmileRight")) * 1.8
    )

    channels["blinkBoth"] = (
        (blink_l + blink_r) * 0.5
        if abs(blink_l - blink_r) < 0.25
        else 0.0
    )

    channels["winkLeft"] = (
        blink_l if blink_l > 0.35 and blink_r < 0.25 else 0.0
    )

    channels["winkRight"] = (
        blink_r if blink_r > 0.35 and blink_l < 0.25 else 0.0
    )

    return channels


# =========================================================
# КАЛИБРОВКА
# =========================================================

def record_stage(camera, landmarker, target_name, description):
    cv2.namedWindow("VTuber Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VTuber Calibration", 720, 520)

    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        cv2.rectangle(
            frame, (10, 10), (w - 10, 110), (20, 20, 30), -1
        )

        cv2.putText(
            frame, description, (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (0, 255, 255), 2
        )

        cv2.putText(
            frame,
            "Прими выражение и нажми ПРОБЕЛ (Q - выход)",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
            (255, 255, 255), 1
        )

        cv2.imshow("VTuber Calibration", frame)

        key = cv2.waitKey(20) & 0xFF

        if key == ord(" "):
            break

        if key in (ord("q"), 27):
            return None

    start = time.time()

    while time.time() - start < 2.0:
        ret, frame = camera.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)

        sec = int(2.5 - (time.time() - start))

        cv2.putText(
            frame, f"Держи лицо! Запись через {sec}...",
            (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 165, 255), 3
        )

        cv2.imshow("VTuber Calibration", frame)
        cv2.waitKey(20)

    samples = []

    while len(samples) < 30:
        ret, frame = camera.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        ts = int(time.time() * 1000)
        res = landmarker.detect_for_video(mp_img, ts)

        if res.face_landmarks:
            bs = extract_blendshapes(res)

            if bs:
                samples.append(bs)

        cv2.putText(
            frame, f"Запись: {len(samples)}/30 кадров",
            (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
            1.0, (0, 255, 0), 2
        )

        cv2.imshow("VTuber Calibration", frame)
        cv2.waitKey(15)

    avg_bs = {}

    for name in BLENDSHAPE_NAMES:
        vals = [s.get(name, 0.0) for s in samples]
        avg_bs[name] = float(np.mean(vals)) if vals else 0.0

    return {
        "samples": len(samples),
        "blendshapes": avg_bs
    }


def calibrate(camera, landmarker):
    global calibration

    collected = {}

    print("\n--- КАЛИБРОВКА 12 СОСТОЯНИЙ ---")

    for name, description in TARGETS:
        result = record_stage(
            camera, landmarker, name, description
        )

        if result is None:
            print("Калибровка прервана.")
            return False

        collected[name] = result

    calibration = collected

    with CALIBRATION_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            {"version": 3, "targets": calibration},
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"Калибровка сохранена: {CALIBRATION_PATH}")
    return True


def load_calibration():
    global calibration

    if not CALIBRATION_PATH.exists():
        return False

    try:
        with CALIBRATION_PATH.open("r", encoding="utf-8") as file:
            data = json.load(file)

        calibration = data.get("targets", data)

        print("✓ face_calibration.json загружен.")
        return True

    except Exception as error:
        print(f"Ошибка чтения калибровки: {error}")
        return False


# =========================================================
# WEBSOCKET
# =========================================================

async def broadcast(packet):
    if not clients:
        return

    message = json.dumps(packet, ensure_ascii=False)

    await asyncio.gather(
        *[client.send(message) for client in list(clients)],
        return_exceptions=True
    )


async def ws_handler(ws):
    clients.add(ws)

    print(f"+ Клиент подключён ({len(clients)})")

    await ws.send(json.dumps({
        "type": "status",
        "message": "Python трекер готов."
    }))

    try:
        await ws.wait_closed()
    finally:
        clients.discard(ws)
        print(f"- Клиент отключён ({len(clients)})")


# =========================================================
# ОСНОВНОЙ ЦИКЛ ТРЕКИНГА
# =========================================================

async def tracking_loop(camera, landmarker):
    cv2.namedWindow("VTuber Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VTuber Tracking", 640, 480)

    previous_loop_time = time.perf_counter()

    while True:
        ret, frame = camera.read()

        if not ret:
            await asyncio.sleep(0.01)
            continue

        now = time.perf_counter()
        dt = min(now - previous_loop_time, 0.1)
        previous_loop_time = now

        audio_strength, rms = update_audio_envelope(dt)

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        # VIDEO требует возрастающие timestamp.
        global last_timestamp_ms
        ts = max(
            int(time.monotonic() * 1000),
            last_timestamp_ms + 1
        )
        last_timestamp_ms = ts

        result = landmarker.detect_for_video(mp_img, ts)

        if result.face_landmarks:
            raw_face = extract_blendshapes(result)
            face = smooth_face(raw_face)

            # Звук добавляет движение челюсти.
            combined_jaw, audio_jaw = apply_audio_to_face(
                face, audio_strength
            )

            head = head_rotation(result)

            # Независимые сигналы частей лица.
            channels = compute_face_channels(
                face, audio_strength
            )

            packet = {
                "type": "face",
                "face": face,
                "head": head,
                "signals": channels,
                "audio": {
                    "rms": rms,
                    "strength": audio_strength,
                    "jawContribution": audio_jaw,
                    "combinedJawOpen": combined_jaw
                }
            }

            await broadcast(packet)

            # Диагностика без главной эмоции.
            y = 28

            cv2.putText(
                frame,
                f"Audio: {audio_strength:.2f} | RMS: {rms:.3f}",
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2
            )

            y += 24

            cv2.putText(
                frame,
                f"Jaw: {combined_jaw:.2f} | "
                f"Smile L/R: {face.get('mouthSmileLeft', 0):.2f}/"
                f"{face.get('mouthSmileRight', 0):.2f}",
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1
            )

            y += 22

            cv2.putText(
                frame,
                f"Brows L/R: {channels['browDownLeft']:.2f}/"
                f"{channels['browDownRight']:.2f} | "
                f"Blink L/R: {channels['eyeBlinkLeft']:.2f}/"
                f"{channels['eyeBlinkRight']:.2f}",
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (255, 255, 255), 1
            )

            cv2.putText(
                frame,
                "[C] - перекалибровка | [Q] - выход",
                (15, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (180, 180, 180), 1
            )

        else:
            await broadcast({
                "type": "status",
                "message": "Лицо не обнаружено"
            })

            cv2.putText(
                frame, "Лицо не обнаружено",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 0, 255), 2
            )

        cv2.imshow("VTuber Tracking", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            calibrate(camera, landmarker)

        elif key in (ord("q"), 27):
            break

        await asyncio.sleep(0.001)


# =========================================================
# ЗАПУСК
# =========================================================

async def main():
    if not MODEL_PATH.exists():
        print(f"Ошибка: не найден файл {MODEL_PATH}")
        sys.exit(1)

    camera = cv2.VideoCapture(CAM_INDEX)

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not camera.isOpened():
        print(f"Ошибка: не удалось открыть камеру {CAM_INDEX}")
        sys.exit(1)

    base_opts = mp.tasks.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    )

    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_opts,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1
    )

    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
        opts
    )

    start_audio()

    if not load_calibration():
        print("Калибровка не найдена. Запуск мастера калибровки...")

        if not calibrate(camera, landmarker):
            camera.release()
            cv2.destroyAllWindows()
            return

    server = await websockets.serve(
        ws_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT
    )

    print(
        f"WebSocket: ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}"
    )

    try:
        await tracking_loop(camera, landmarker)

    finally:
        camera.release()
        cv2.destroyAllWindows()

        if audio_stream is not None:
            audio_stream.stop()
            audio_stream.close()

        server.close()
        await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nОстановка трекера.")
