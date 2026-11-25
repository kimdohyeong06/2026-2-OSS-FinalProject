import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def load_resized_image(img_path, target_w=1280, target_h=720):
    """원본 가로·세로 비율을 유지하면서 1280x720 화면 중앙에 배치하는 함수"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((target_h, target_w, 3), 30, dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas

def detect_ppe_on_image(model, img_path):
    """이미지를 불러와 비율을 맞춘 뒤 YOLO 모델로 판별하고 UI를 그리는 함수"""
    frame = load_resized_image(img_path)
    if frame is None:
        return np.full((720, 1280, 3), 40, dtype=np.uint8)

    # YOLO 모델 추론 실행
    results = model(frame, conf=0.5, verbose=False)
    has_warning = False

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # 미착용 판별 ('no' 또는 'without' 포함 시 경고)[cite: 4]
            if "no" in class_name.lower() or "without" in class_name.lower():
                has_warning = True
                color = (0, 0, 255)  # 빨간색[cite: 4]
                label = f"WARNING: {class_name} ({conf:.2f})"
            else:
                color = (0, 255, 0)  # 초록색[cite: 4]
                label = f"{class_name} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 미착용 감지 시 상단 경고 배너 출력[cite: 4]
    if has_warning:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
        cv2.putText(frame, "PPE CHECK WARNING! NO HELMET DETECTED", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame

if __name__ == "__main__":
    # 1. 학습 완료 후 받을 모델 경로 & 테스트할 이미지 폴더 경로[cite: 4, 5]
    MODEL_PATH = "models/helmet_yolo26.pt"
    IMAGE_DIR = Path("stream/images")

    model = YOLO(MODEL_PATH)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in valid_exts] if IMAGE_DIR.exists() else []

    if not image_files:
        print(f"[Error] '{IMAGE_DIR}' 폴더에 이미지가 없습니다.")
    else:
        current_idx = 0
        # 첫 번째 이미지 AI 판별 실행
        result_frame = detect_ppe_on_image(model, image_files[current_idx])

        print("=== AI 모델 기반 다중 이미지 판별 시작 ===")
        print(" - 키보드 'n' : 다음 이미지 판별")
        print(" - 키보드 'p' : 이전 이미지 판별")
        print(" - 키보드 's' : 현재 판별 결과 이미지로 저장")
        print(" - 키보드 'q' : 종료")

        while True:
            display = result_frame.copy()
            file_label = f"[{current_idx + 1}/{len(image_files)}] {image_files[current_idx].name}"
            cv2.putText(display, f"{file_label} | 'n': Next, 'p': Prev, 's': Save, 'q': Quit",
                        (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            cv2.imshow("Image PPE Detection", display)

            key = cv2.waitKey(30) & 0xFF
            if key == ord('n'):
                current_idx = (current_idx + 1) % len(image_files)
                result_frame = detect_ppe_on_image(model, image_files[current_idx])
            elif key == ord('p'):
                current_idx = (current_idx - 1) % len(image_files)
                result_frame = detect_ppe_on_image(model, image_files[current_idx])
            elif key == ord('s'):
                save_name = f"result_{image_files[current_idx].name}"
                cv2.imwrite(save_name, result_frame)
                print(f"저장 완료: {save_name}")
            elif key == ord('q'):
                break

        cv2.destroyAllWindows()