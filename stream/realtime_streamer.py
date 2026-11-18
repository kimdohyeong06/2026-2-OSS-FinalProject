import cv2
import time
from ultralytics import YOLO
from stream.stream import RealtimeStreamer  # 작성하신 stream_2.py에서 클래스 불러오기[cite: 10]

# 1. train.py로 학습 완료된 최적 가중치(best.pt) 로드
# 기본 설정 기준 저장 경로: runs/detect/helmet_v1/weights/best.pt
MODEL_PATH = "runs/detect/helmet_v1/weights/best.pt"
model = YOLO(MODEL_PATH)

# 2. 스트리머 객체 생성 및 카메라 시작[cite: 10]
streamer = RealtimeStreamer(cam_id=0)

if streamer.start():
    print("=== 학습된 안전모 탐지 모델 실시간 모니터링 시작 ('q' 누르면 종료) ===")
    
    while True:
        ret, frame = streamer.get_frame()  #[cite: 10]
        if not ret:
            break

        # 3. 실시간 프레임에 학습된 YOLO 모델 추론 실행
        results = model(frame, stream=True, conf=0.5, verbose=False)
        
        has_warning = False

        for r in results:
            for box in r.boxes:
                # 바운딩 박스 좌표, 신뢰도, 클래스 이름 추출
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = model.names[cls_id]  # data.yaml에 정의된 클래스명 출력

                # 4. 미착용 여부 판별 및 시각화 (데이터셋 클래스명에 맞게 'no' 또는 'without' 포함 여부 체크)
                if "no" in class_name.lower() or "without" in class_name.lower():
                    has_warning = True
                    color = (0, 0, 255)  # 미착용: 빨간색
                    label = f"WARNING: {class_name} ({conf:.2f})"
                else:
                    color = (0, 255, 0)  # 정상 착용: 초록색
                    label = f"{class_name} ({conf:.2f})"

                # 프레임에 박스와 라벨 그리기
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 5. 미착용 감지 시 상단 경고 배너 출력
        if has_warning:
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 255), -1)
            cv2.putText(frame, "PPE CHECK WARNING! NO HELMET DETECTED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Realtime PPE Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    streamer.stop()  #[cite: 10]