# Hand-Eye-calibration
## 목표: 카메라에 보이는 객체의 중심점을 robot base frame 기준 6D 좌표로 변환

1. 0910 진행 상황
- AE/AWB (자동노출/화이트밸런스)를 lock(프리셋 적용)하는 코드 작성/확인.
- 캘리브레이션 보드: asymmetric circle grid, `findCirclesGrid()`로 검출 성공(20개 포인트)
  <img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/7c8af07c-c7ba-4432-b60a-ad4556131478" />  
- Realsense에서 1280×720 인트린식(K, D)을 추출
- 한 프레임에 대해 undistort → patch-refine → solvePnP → reproj RMSE ≈ 0.07 px(다른 프레임으로 검증 필요)
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/a519c86d-b4b4-4fce-92e0-f5019c81c5df" />

- 2D 검출은 OK, 일부 depth 값이 이상치(예: 1.5m이상 등) 로 들어있어 depth 보정 필요함
<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0caf752e-85ae-4559-a2f2-000dff5e24e1" />
