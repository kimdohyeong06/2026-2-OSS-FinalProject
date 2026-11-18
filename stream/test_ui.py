import cv2
import numpy as np
from pathlib import Path

def load_resized_image(img_path, target_w=1280, target_h=720):
    """원본 가로·세로 비율을 유지하면서 1280x720 화면 중앙에 배치하는 함수"""
    img = cv2.imread(str(img_path))
    if img is None:
        return np.full((target_h, target_w, 3), 40, dtype=np.uint8)

    h, w = img.shape[:2]
    
    # 가로와 세로 중 목표 크기에 맞추기 위한 더 작은 축소/확대 비율 계산
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # 원본 비율을 유지한 채 크기 조절
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 1280x720 크기의 어두운 배경 캔버스 생성
    canvas = np.full((target_h, target_w, 3), 30, dtype=np.uint8)

    # 이미지를 캔버스 정중앙에 배치할 시작 좌표 계산
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas

if __name__ == "__main__":
    # 1. 이미지들이 들어있는 폴더 경로 지정 (원하는 폴더 경로로 수정 가능)
    IMAGE_DIR = Path("stream/images")
    
    # 폴더 내 jpg, jpeg, png 파일 목록 모두 가져오기
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [p for p in IMAGE_DIR.glob("*") if p.suffix.lower() in valid_exts] if IMAGE_DIR.exists() else []

    if not image_files:
        print(f"[Warning] '{IMAGE_DIR}' 폴더가 없거나 이미지가 없습니다. 기본 회색 화면으로 실행합니다.")
        image_files = [None]

    current_idx = 0
    base_img = load_resized_image(image_files[current_idx]) if image_files[0] else np.full((720, 1280, 3), 40, dtype=np.uint8)

    print("=== 다중 이미지 UI/경고 테스트 시작 ===")
    print(" - 키보드 'n' : 다음 이미지 (Next)")
    print(" - 키보드 'p' : 이전 이미지 (Prev)")
    print(" - 키보드 'w' : 미착용 경고 강제 발생 / 해제")
    print(" - 키보드 'q' : 테스트 종료")

    is_warning_mode = False

    while True:
        frame = base_img.copy()

        # 현재 보고 있는 파일 이름 상단 표시
        file_label = f"[{current_idx + 1}/{len(image_files)}] {image_files[current_idx].name}" if image_files[0] else "[Virtual Screen]"

        # [기본 UI] 정상 착용자 가상 박스 (초록색)
        cv2.rectangle(frame, (100, 150), (300, 450), (0, 255, 0), 2)
        cv2.putText(frame, "HELMET (0.95)", (100, 140), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # [경고 UI] 'w' 키를 눌렀을 때만 빨간색 경고 표시
        if is_warning_mode:
            cv2.rectangle(frame, (450, 150), (650, 450), (0, 0, 255), 2)
            cv2.putText(frame, "WARNING: NO-HELMET (0.88)", (450, 140), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, "PPE CHECK WARNING! UNAPPROVED ACCESS DETECTED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 하단 조작 안내 및 현재 파일명 출력
        cv2.putText(frame, f"{file_label} | 'n': Next, 'p': Prev, 'w': Warning, 'q': Quit", 
                    (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("UI & Warning Test (Multi-Image)", frame)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('n') and image_files[0]:
            # 다음 이미지로 이동
            current_idx = (current_idx + 1) % len(image_files)
            base_img = load_resized_image(image_files[current_idx])
        elif key == ord('p') and image_files[0]:
            # 이전 이미지로 이동
            current_idx = (current_idx - 1) % len(image_files)
            base_img = load_resized_image(image_files[current_idx])
        elif key == ord('w'):
            is_warning_mode = not is_warning_mode
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()