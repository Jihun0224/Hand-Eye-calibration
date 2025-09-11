# Hand-Eye-calibration
## 목표: 카메라에 보이는 객체의 중심점을 robot base frame 기준 6D 좌표로 변환
### 현재 진행 상황
- 데이터 수집 파이프라인 완성
  - 프레임 저장: calib_images/he_img_XXXX.png
  - 프레임 메타/로봇 상태 저장: read_positions/he_record_XXXX.json (gripper_pose_base H, joints, board_detection)
- 보드/원 검출 코드와 디버그 시각화 스크립트 작성
- 각 포인트 인덱스 · depth 텍스트 생성 기능

- 문제 발견 및 진단
  - 많은 프레임에서 PnP 재투영 오차가 매우 큼
  - 보드 검출이 일부 이미지에서 실패 — 여러 전처리/탐지 방식을 시도

### 남은 작업
1) 카메라 인트린식 정리 (최우선)
  - 목적: 정확한 K와 D 획득
2) 보드/타겟 검출 안정화
  - 목적: 이미지에서 일관되게 점을 검출
3) Hand–Eye 캘리브레이션 실행
  - 목적: camera ↔ gripper (또는 gripper ↔ camera) rigid transform 계산
  - cv2.calibrateHandEye 또는 등가 함수로 R_cam2grip, t_cam2grip 산출.
4) 픽셀→p_cam→p_base 변환 파이프라인 완성
  - 목적: 단일 검출점(u,v) + depth → p_base
5) 목표 pose 생성 및 모션 실행
  - 목적: 계산된 p_base로 왼팔 이동
6) 정밀화 및 검증


---
1. 0910 진행 상황
- AE/AWB (자동노출/화이트밸런스)를 lock(프리셋 적용)하는 코드 작성/확인.
- 캘리브레이션 보드: asymmetric circle grid, `findCirclesGrid()`로 검출 성공(20개 포인트)
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/7c8af07c-c7ba-4432-b60a-ad4556131478" />  
- Realsense에서 1280×720 인트린식(K, D)을 추출
- 한 프레임에 대해 undistort → patch-refine → solvePnP → reproj RMSE ≈ 0.07 px(다른 프레임으로 검증 필요)
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/a519c86d-b4b4-4fce-92e0-f5019c81c5df" />

- 2D 검출은 OK, 일부 depth 값이 이상치(예: 1.5m이상 등) 로 들어있어 depth 보정 필요함
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0caf752e-85ae-4559-a2f2-000dff5e24e1" />
