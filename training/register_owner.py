from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import torch
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)
# ======== CONSTANTS ========
EMBEDDING_PATH = os.path.join(DATA_DIR, 'owner_embedding.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ======== MODELS ========
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ======== VIDEO CAPTURE ========
cap = cv2.VideoCapture(0)
embeddings = []

print("Nhấn 's' để chụp khuôn mặt.")
print("Nhấn 'q' để kết thúc và lưu owner.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = mtcnn.detect(img_rgb)

    # Vẽ bounding box nếu phát hiện
    boxes = None
    if isinstance(result, tuple) and len(result) >= 2:
        boxes = result[0]

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Register Owner', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        # Chụp và tính embedding
        face = mtcnn(img_rgb)
        if face is not None:
            with torch.no_grad():
                emb = resnet(face.unsqueeze(0).to(device))
                embeddings.append(emb.cpu().numpy())
            print("Da luu 1 embedding.")
        else:
            print("Khong phat hien khuon mat!")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# ======== SAVE EMBEDDING ========
if embeddings:
    owner_embedding = np.mean(np.vstack(embeddings), axis=0)
    torch.save(owner_embedding, EMBEDDING_PATH)
    print(f"Đã lưu embedding chủ xe vào: {EMBEDDING_PATH}")
else:
    print("Không có embedding nào được lưu.")
