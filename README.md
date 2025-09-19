# Hand-Eye-calibration
## 목표: 카메라에 보이는 객체의 중심점을 robot base frame 기준 6D 좌표로 변환
###  Hand–Eye Calibration & Pixel→Base 이동 파이프라인 (2025-09-19)

로봇 머리 카메라(Realsense)와 로봇 베이스/토르소/그리퍼 좌표계를 정합하여, 카메라 2D(픽셀) → 3D(카메라) → 토르소5 → 로봇 이동까지 끝-to-끝(E2E)로 동작
1. 요약
  - 데이터셋 평가: keep=21/25, reproj ≈ 0.03 px, depth residual ≈ 8.6 mm
  - Hand–Eye 추정 (NLS): 보드 포인트 base 정합 RMSE ≈ 2.13 mm (median ≈ 1.90 mm)
  - H_link_torso5_cam 생성(p_cam → p_torso5)  

2. 디렉토리/파일 개요

  - handeye_dataset_eval.py  
수집 JSON(calib_records/*.json) 일괄 평가/통계/CSV (handeye_dataset_eval.csv)


- drop_in_solve_handeye_nls.py (NLS hand–eye 솔버심)  
비선형 최소제곱으로 ^baseH_cam, ^gripH_board 동시 최적화. RMSE≈2mm.

- save_torso5_cam_transform.py  
운영용 고정 변환 H_link_torso5_cam = inv(^baseH_torso5) · ^baseH_cam 계산/저장 → handeye_result_fixed.json.

- viz_axes_one.py, viz_axes_batch.py  
단일/일괄 축 가시화(카메라/보드/그리퍼/베이스).



1. 0910 진행 상황
- AE/AWB (자동노출/화이트밸런스)를 lock(프리셋 적용)하는 코드 작성/확인.
- 캘리브레이션 보드: asymmetric circle grid, `findCirclesGrid()`로 검출 성공(20개 포인트)
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/7c8af07c-c7ba-4432-b60a-ad4556131478" />  
- Realsense에서 1280×720 인트린식(K, D)을 추출
- 한 프레임에 대해 undistort → patch-refine → solvePnP → reproj RMSE ≈ 0.07 px(다른 프레임으로 검증 필요)
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/a519c86d-b4b4-4fce-92e0-f5019c81c5df" />

- 2D 검출은 OK, 일부 depth 값이 이상치(예: 1.5m이상 등) 로 들어있어 depth 보정 필요함
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0caf752e-85ae-4559-a2f2-000dff5e24e1" />

--> depth 측정시 60cm 이상의 거리에서 측정 필요, 해상도에 따른 K 설정 확인