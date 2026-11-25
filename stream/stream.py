import cv2
from ultralytics import YOLO

class RealtimeStreamer:
    def __init__(self, cam_id=0, model_path="models/helmet_yolo26.pt", width=1280, height=720):
        self.cam_id = cam_id
        self.model_path = model_path
        self.width = width
        self.height = height
        self.cap = None
        self.is_running = False
        self.model = None

    def load_model(self):
        """YOLO 모델 로드"""
        self.model = YOLO(self.model_path)

    def start(self):
        """카메라 스트림 시작"""
        self.cap = cv2.VideoCapture(self.cam_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self.cap.isOpened():
            return False

        self.is_running = True
        return True

    def get_frame(self):
        """실시간 프레임 읽기"""
        if not self.is_running or self.cap is None:
            return False, None
        return self.cap.read()

    def detect(self, frame, conf=0.5):
        """
        단일 프레임에 대해 안전모 착용/미착용 식별만 수행
        반환값: (시각화된 프레임, 착용 인원, 미착용 인원, 미착용자 바운딩박스 리스트)
        """
        results = self.model(frame, conf=conf, verbose=False)
        annotated_frame = results[0].plot()

        wearing = 0
        not_wearing = 0
        violator_boxes = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = self.model.names[cls_id].lower()
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 'no_helmet' 또는 'without' 계열 클래스 -> 미착용
            if "no" in name or "without" in name or name == "head":
                not_wearing += 1
                violator_boxes.append((x1, y1, x2, y2))
            # 'helmet' 계열 클래스 -> 착용 ('person' 단독 클래스는 카운트에서 제외하여 중복 방지)
            elif "helmet" in name or "hardhat" in name:
                wearing += 1

        return annotated_frame, wearing, not_wearing, violator_boxes

    def stop(self):
        """카메라 자원 해제 (GUI 환경이므로 cv2.destroyAllWindows()는 제거)"""
        self.is_running = False
        if self.cap and self.cap.isOpened():
            self.cap.release()