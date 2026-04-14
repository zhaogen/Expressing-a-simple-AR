import cv2 as cv
import numpy as np

# 1. 카메라 파라미터 설정 (본인의 HW3 결과값으로 반드시 수정하세요!)
K = np.array([[1.0e+03, 0.0, 9.6e+02], 
              [0.0, 1.0e+03, 5.4e+02], 
              [0.0, 0.0, 1.0]], dtype=np.float32)
dist = np.zeros(5, dtype=np.float32) # [k1, k2, p1, p2, k3]

# 2. 체스판 설정 (영상 기준 코너 개수)
board_pattern = (7, 4) 
square_size = 1.0 # 실제 칸 크기(cm/mm)를 넣으면 실제 크기로 그려집니다.

# 3. 3D 물체 좌표 정의 (정육면체)
# 체스판 평면(z=0) 위에 한 변의 길이가 2인 정육면체를 정의합니다.
obj_pts = np.float32([[0,0,0], [2,0,0], [2,2,0], [0,2,0],
                      [0,0,-2], [2,0,-2], [2,2,-2], [0,2,-2]])

# 4. 체스판의 3D 코너 좌표 미리 계산
objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2) * square_size

# 영상 읽기
video_path = 'Checkboardvideo.mp4'
cap = cv.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # 5. 체스판 코너 찾기
    found, corners = cv.findChessboardCorners(gray, board_pattern, None)

    if found:
        # 자세 추정 (Pose Estimation)
        _, rvec, tvec = cv.solvePnP(objp, corners, K, dist)

        # 3D 점들을 2D 이미지 평면으로 투영 [cite: 2]
        img_pts, _ = cv.projectPoints(obj_pts, rvec, tvec, K, dist)
        img_pts = np.int32(img_pts).reshape(-1, 2)

        # 6. 정육면체 그리기
        # 바닥면 (빨간색)
        cv.drawContours(frame, [img_pts[:4]], -1, (0, 0, 255), 2)
        # 기둥 (초록색)
        for i, j in zip(range(4), range(4, 8)):
            cv.line(frame, tuple(img_pts[i]), tuple(img_pts[j]), (0, 255, 0), 2)
        # 윗면 (파란색)
        cv.drawContours(frame, [img_pts[4:]], -1, (255, 0, 0), 2)

    cv.imshow('AR Cube Assignment', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()