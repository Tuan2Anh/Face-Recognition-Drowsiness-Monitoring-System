from modules import recognize_owner
from modules import monitor_drowsiness
import serial
import time
import cv2
import torch
import os
import numpy as np
from facenet_pytorch import InceptionResnetV1

# ======== Cấu hình ========
SERIAL_PORT = 'COM3'
BAUDRATE = 9600
MAX_WRONG_ATTEMPTS = 3
SIM_THRESHOLD = 0.7

# ======== Load embedding chủ xe ========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EMBEDDING_PATH = os.path.join(BASE_DIR, 'data', 'owner_embedding.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if not os.path.exists(EMBEDDING_PATH):
    print(f"Khong tim thay file: {EMBEDDING_PATH}")
    print("Vui long chay register_owner.py de dang ky khuon mat truoc.")
    exit()

state = torch.load(EMBEDDING_PATH, map_location=device)
if isinstance(state, torch.Tensor):
    owner_embedding = state.unsqueeze(0)
elif isinstance(state, np.ndarray):
    owner_embedding = torch.from_numpy(state).unsqueeze(0)
else:
    raise TypeError(f"Dang du lieu khong hop le: {type(state)}")

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def send_command(ser, command):
    try:
        ser.write(command.encode())
        print(f"Gui lenh: {command}")
    except Exception as e:
        print(f"Loi gui lenh: {e}")

def wait_for_button(ser, cap, wrong_attempts):
    print("Khoi dong webcam... Dua mat vao khung va nhan nut de bat dau nhan dien.")
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        face_found = recognize_owner.has_face(frame)

        msg = "Phat hien khuon mat. Nhan nut de nhan dien..." if face_found \
              else "Khong tim thay khuon mat"
        color = (0, 255, 0) if face_found else (0, 0, 255)

        cv2.putText(frame, msg, (30, 40), font, 0.8, color, 2)
        cv2.putText(frame, f"Lan sai: {wrong_attempts}", (30, 400), font, 0.7, (0, 0, 255), 2)

        cv2.imshow("Camera - Nhan nut de bat dau", frame)

        if ser.in_waiting:
            line = ser.readline().decode().strip()
            if line == "BUTTON_PRESSED":
                print("Nut duoc nhan, bat dau kiem tra nhan dien...")
                return frame

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def main():
    print("He thong khoi dong...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)
        time.sleep(2)
    except Exception as e:
        print(f"Khong the mo cong COM: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Khong the mo camera.")
        ser.close()
        return

    wrong_attempts = 0

    while True:
        frame = wait_for_button(ser, cap, wrong_attempts)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = recognize_owner.face_detector.detect(rgb)

        is_owner = False
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                face_tensor = recognize_owner.face_detector.extract(rgb, [box], save_path=None)
                if face_tensor is not None and len(face_tensor) > 0:
                    with torch.no_grad():
                        emb = resnet(face_tensor[0].unsqueeze(0).to(device))
                        sim = torch.nn.functional.cosine_similarity(owner_embedding, emb).item()
                        print(f"Similarity: {sim:.2f}")
                        if sim > SIM_THRESHOLD:
                            is_owner = True
                            break

        if is_owner:
            print("Xac nhan dung chu xe.")
            send_command(ser, 'G')  # Bật đèn xanh
            break
        else:
            wrong_attempts += 1
            print(f"Khong phai chu xe. Lan sai thu {wrong_attempts}")
            send_command(ser, 'R')  # Bật đèn đỏ
            time.sleep(3)

            if wrong_attempts >= MAX_WRONG_ATTEMPTS:
                print("Sai qua 3 lan! Bat coi canh bao.")
                send_command(ser, 'B')  # Bật cảnh báo
                time.sleep(3)
                send_command(ser, 'S')  # Tắt cảnh báo
                wrong_attempts = 0

    cap.release()
    cv2.destroyAllWindows()

    print("Bat dau giam sat buon ngu...")
    monitor_drowsiness.monitor_drowsiness(ser)

    ser.close()
    print("He thong ket thuc.")

if __name__ == "__main__":
    main()