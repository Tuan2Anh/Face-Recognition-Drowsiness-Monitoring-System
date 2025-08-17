import os
import cv2
import time
import math
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque

def monitor_drowsiness(ser):  # ✅ Truyền 'ser' vào thay vì tạo mới
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(BASE_DIR, '..', 'data', 'eye_state_model.h5')

    model = tf.keras.models.load_model(model_path)

    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    drawing_spec = mp_drawing.DrawingSpec(
        color=(0, 255, 0), thickness=1, circle_radius=1)

    LEFT_EYE_LMS = [33, 133, 160, 159, 158, 145, 153, 154, 155]
    RIGHT_EYE_LMS = [362, 263, 387, 386, 385, 373, 380, 381, 382]

    EYE_CLOSED_SEC = 2.0
    closed_start = None

    BLINK_MIN_SEC = 0.07
    BLINK_MAX_SEC = 0.30
    blink_cnt = 0

    perclos_win = deque(maxlen=60 * 30)
    perclos_thres = 0.4
    STAT_PERIOD = 0.3
    last_stat_t = time.time()

    HEAD_PITCH_THRES = 8
    no_face_cnt, NO_FACE_FRAMES = 0, 30

    fps, prev_time, FPS_SMOOTH = 0.0, time.time(), 0.8

    PNP = {"nose": 1, "chin": 152, "l_eye": 33,
           "r_eye": 263, "l_mouth": 61, "r_mouth": 291}
    MODEL_PTS = np.float32([
        (0, 0, 0), (0, -330, -65), (-225, -170, -135), (225, -170, -135),
        (-150, -150, -125), (150, -150, -125)
    ])

    clahe = cv2.createCLAHE(2.0, (8, 8))

    def crop_eye_by_lms(frame, lm, idxs, margin=5):
        h, w, _ = frame.shape
        xs = [lm[i].x * w for i in idxs]
        ys = [lm[i].y * h for i in idxs]
        x1, x2 = int(max(min(xs) - margin, 0)), int(min(max(xs) + margin, w))
        y1, y2 = int(max(min(ys) - margin, 0)), int(min(max(ys) + margin, h))
        return frame[y1:y2, x1:x2]

    def pre_eye(img):
        if img.size == 0:
            return None
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = clahe.apply(g)
        g = cv2.resize(g, (24, 24))
        g = g / 255.0
        return g.reshape(1, 24, 24, 1)

    def mar(lm):
        v = np.hypot(lm[13].x - lm[14].x, lm[13].y - lm[14].y)
        h = np.hypot(lm[78].x - lm[308].x, lm[78].y - lm[308].y)
        return v / h if h > 0 else 0

    def head_pitch(lm, size):
        h, w = size
        pts = np.float32([
            (lm[PNP["nose"]].x * w, lm[PNP["nose"]].y * h),
            (lm[PNP["chin"]].x * w, lm[PNP["chin"]].y * h),
            (lm[PNP["l_eye"]].x * w, lm[PNP["l_eye"]].y * h),
            (lm[PNP["r_eye"]].x * w, lm[PNP["r_eye"]].y * h),
            (lm[PNP["l_mouth"]].x * w, lm[PNP["l_mouth"]].y * h),
            (lm[PNP["r_mouth"]].x * w, lm[PNP["r_mouth"]].y * h)])
        cam = np.array([[w, 0, w / 2], [0, w, h / 2], [0, 0, 1]], np.float32)
        ok, rvec, _ = cv2.solvePnP(MODEL_PTS, pts, cam, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return 0.0
        rot, _ = cv2.Rodrigues(rvec)
        return math.degrees(math.atan2(-rot[2][0], (rot[2][1] ** 2 + rot[2][2] ** 2) ** 0.5))

    cap = cv2.VideoCapture(0)

    drowsy_alert_sent = False
    head_nod_start_time = None
    buzzer_on = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        now = time.time()
        inst_fps = 1.0 / (now - prev_time) if now != prev_time else 0
        fps = FPS_SMOOTH * fps + (1 - FPS_SMOOTH) * inst_fps
        prev_time = now

        res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if not res.multi_face_landmarks:
            no_face_cnt += 1
            if no_face_cnt > NO_FACE_FRAMES:
                cv2.putText(frame, "No Driver Detected!", (30, 360),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            no_face_cnt = 0
            for fl in res.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, fl, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

                lm = fl.landmark
                L_roi = crop_eye_by_lms(frame, lm, LEFT_EYE_LMS)
                R_roi = crop_eye_by_lms(frame, lm, RIGHT_EYE_LMS)

                L_in, R_in = pre_eye(L_roi), pre_eye(R_roi)
                if L_in is None or R_in is None:
                    continue

                eye_state = (model.predict(L_in)[0][0] + model.predict(R_in)[0][0]) / 2

                if eye_state < 0.5:
                    label = "Eyes Closed"
                    if closed_start is None:
                        closed_start = time.time()
                    perclos_win.append(1)

                    if time.time() - closed_start >= EYE_CLOSED_SEC:
                        cv2.putText(frame, "DROWSINESS ALERT!", (30, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        if not drowsy_alert_sent:
                            try:
                                ser.write(b'B')  # thay arduino bằng ser
                                drowsy_alert_sent = True
                            except:
                                pass
                    else:
                        if drowsy_alert_sent:
                            try:
                                ser.write(b'S')
                                drowsy_alert_sent = False
                            except:
                                pass
                else:
                    label = "Eyes Open"
                    if closed_start:
                        dur = time.time() - closed_start
                        if BLINK_MIN_SEC <= dur <= BLINK_MAX_SEC:
                            blink_cnt += 1
                    closed_start = None
                    perclos_win.append(0)
                    
                    # Tắt còi khi mắt mở lại
                    if drowsy_alert_sent:
                        try:
                            ser.write(b'S')
                            drowsy_alert_sent = False
                        except:
                            pass

                cv2.putText(frame, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                            (0, 0, 255) if label == "Eyes Closed" else (0, 255, 0), 3)

                if now - last_stat_t > STAT_PERIOD:
                    last_stat_t = now
                    if perclos_win and sum(perclos_win) / len(perclos_win) > perclos_thres:
                        cv2.putText(frame, "Drowsiness Risk (PERCLOS)", (30, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                    pitch = head_pitch(lm, (h, w))
                    if pitch > HEAD_PITCH_THRES:
                        if head_nod_start_time is None:
                            head_nod_start_time = now
                        elif now - head_nod_start_time > 5:
                            cv2.putText(frame, "Head Nodding Detected!", (30, 300),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                            if not buzzer_on:
                                try:
                                    ser.write(b'B')
                                    buzzer_on = True
                                except:
                                    pass
                    else:
                        head_nod_start_time = None
                        if buzzer_on:
                            try:
                                ser.write(b'S')
                                buzzer_on = False
                            except:
                                pass

                if mar(lm) > 0.6:
                    cv2.putText(frame, "Yawning Detected", (30, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        # Kiểm tra và tắt còi nếu không có cảnh báo nào
        if not drowsy_alert_sent and not buzzer_on:
            # Đảm bảo còi đã tắt
            try:
                ser.write(b'S')
            except:
                pass

        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_cnt}", (w - 200, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Driver Monitoring", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import serial
    ser = serial.Serial('COM5', 9600, timeout=1)
    monitor_drowsiness(ser)
    ser.close()
