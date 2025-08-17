from facenet_pytorch import MTCNN as FPT_MTCNN, InceptionResnetV1
import cv2
import torch
import numpy as np
import time
import os


# ======= Phát hiện khuôn mặt (mới thêm) =======
from facenet_pytorch import MTCNN as FPT_MTCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
face_detector = FPT_MTCNN(keep_all=True, device=device)

def has_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = face_detector.detect(rgb)

    # Nếu không có khuôn mặt thì return False
    if boxes is None or len(boxes) == 0:
        return False

    # Load embedding chủ xe
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    EMBEDDING_PATH = os.path.join(DATA_DIR, 'owner_embedding.pt')

    state = torch.load(EMBEDDING_PATH, map_location=device)
    if isinstance(state, torch.Tensor):
        owner_embedding = state.unsqueeze(0)
    elif isinstance(state, np.ndarray):
        owner_embedding = torch.from_numpy(state).unsqueeze(0)
    else:
        raise TypeError(f"Unexpected type in {EMBEDDING_PATH}: {type(state)}")

    # Mô hình nhận diện khuôn mặt
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    for box in boxes:
        x1, y1, x2, y2 = [int(x) for x in box]

        face_tensor = face_detector.extract(rgb, [box], save_path=None)
        if face_tensor is not None and len(face_tensor) > 0:
            with torch.no_grad():
                emb = resnet(face_tensor[0].unsqueeze(0).to(device))
                sim = torch.nn.functional.cosine_similarity(owner_embedding, emb).item()
            if sim > 0.7:
                label = f'OWNER ({sim:.2f})'
                color = (0, 255, 0)
            else:
                label = f'NOT OWNER ({sim:.2f})'
                color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return True

def wait_for_face(cap):
    """Hiển thị video đến khi phát hiện khuôn mặt rồi trả về frame."""
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        if has_face(frame):
            print(" Đã phát hiện khuôn mặt! Bấm nút để xác nhận.")
            return frame
        cv2.putText(frame, "Đưa mặt vào camera...", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Phát hiện khuôn mặt", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

# ======= Nhận diện chủ xe =======
def check_owner(ser):
    # ======== ĐƯỜNG DẪN ========
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
    EMBEDDING_PATH = os.path.join(DATA_DIR, 'owner_embedding.pt')
    SIM_THRESHOLD = 0.7  # Có thể điều chỉnh

    # Khởi tạo thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = FPT_MTCNN(image_size=160, margin=20, device=device, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # Load vector khuôn mặt chủ xe
    state = torch.load(EMBEDDING_PATH, weights_only=False, map_location=device)
    if isinstance(state, torch.Tensor):
        owner_embedding = state.unsqueeze(0)
    elif isinstance(state, np.ndarray):
        owner_embedding = torch.from_numpy(state).unsqueeze(0)
    else:
        raise TypeError(f"Unexpected type in {EMBEDDING_PATH}: {type(state)}")

    cap = cv2.VideoCapture(0)
    print("Nhấn 'q' để thoát")
    time.sleep(1)

    owner_confirmed = False
    not_owner_cooldown = 0
    COOLDOWN_FRAMES = 15
    buzzer_on = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mtcnn.detect(img_rgb)

        boxes = None
        if isinstance(result, tuple) and len(result) == 3:
            boxes, probs, _ = result
        else:
            boxes, probs = result

        detected_owner = False
        detected_face = False

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = [int(b) for b in box]
                face = mtcnn.extract(img_rgb, [box], save_path=None)
                if face is not None and len(face) > 0:
                    detected_face = True
                    with torch.no_grad():
                        emb = resnet(face[0].unsqueeze(0).to(device))
                        sim = torch.nn.functional.cosine_similarity(owner_embedding, emb).item()
                        if sim > SIM_THRESHOLD:
                            label = f'OWNER ({sim:.2f})'
                            color = (0, 255, 0)
                            detected_owner = True
                        else:
                            label = f'NOT OWNER ({sim:.2f})'
                            color = (0, 0, 255)
                            if not_owner_cooldown == 0:
                                ser.write(b'B')  # Bật còi
                                not_owner_cooldown = COOLDOWN_FRAMES
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if detected_owner:
                owner_confirmed = True
                if buzzer_on:
                    ser.write(b'S')  # Tắt còi
                    buzzer_on = False
                break
            elif detected_face:
                if not buzzer_on:
                    ser.write(b'B')
                    buzzer_on = True
        else:
            cv2.putText(frame, "Không phát hiện khuôn mặt", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if not buzzer_on:
                ser.write(b'B')
                buzzer_on = True

        if not_owner_cooldown > 0:
            not_owner_cooldown -= 1

        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return owner_confirmed
