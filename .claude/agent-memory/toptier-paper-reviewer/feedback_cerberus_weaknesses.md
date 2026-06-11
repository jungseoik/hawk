---
name: feedback-cerberus-weaknesses
description: CERBERUS 논문에서 반복적으로 나타나는 약점 패턴과 발생 섹션
metadata:
  type: feedback
---

CERBERUS 원고 검토 시 우선 점검할 반복 약점 (2026-06-10 1차 검토 기준).

**약점 패턴:**
- **"무손실 상보 분할" 과대주장:** `(1-M)⊙x`는 정적 영역 *픽셀*은 보존하나 움직이는 객체가 가렸던 배경은 복원 못함(occlusion). 또 motion_mask는 인접프레임 정적객체(주차된 차)를 정적으로 보내므로 의미상 "정적 장면"이 아님. abstract/intro/method 전반에서 "무손실"을 무비판적으로 반복. → "픽셀 분할의 무손실"과 "장면 의미의 무손실"을 구분해야.
- **C5 진단도구가 trivial하게 높아질 위험:** CDS의 Coverage는 z_a 자체가 풍부하면 trivially 높고, Redundancy는 두 인코더가 무관한 노이즈만 학습해도 낮아짐(분리 없이도). → CDS만으로 "분리"를 주장하면 confound. Section 4.4.
- **BSI confound:** 배경교체 시 응답변화가 "정적스트림 인과기여"인지 "외형스트림이 배경픽셀을 본 것"인지 구분 안 됨(외형스트림 X_a는 전체프레임=배경 포함). Section 4.5/4.2. → Dynamic-only 베이스라인도 X_a에 배경이 들어있어 BSI>0 가능.
- **언어손실 L_ML/L_BL ↔ 코드 loss_motion/loss_background 표기 단절:** 논문 수식10/8과 코드(base_task.py)의 대응이 명시 안 됨. 게다가 train_flag==0 경로에선 loss_motion=loss_background=loss(복제). 정합성 구멍. (상세는 [[project-cerberus-code-facts]])
- **L_sim 정당화 약함:** "동적은 외형의 부분집합이므로 z_a≈z_m이어야" → cos→1 강제는 z_a를 동적영역으로 끌어당겨 외형의 정적정보를 훼손할 수 있음(상보성과 충돌). 비대칭 설계의 정보이론 주장이 손으로 흔드는(hand-wavy) 수준, I(z;y) 보존은 코드상 보장장치 없음. Section 3.1.3.
- **plug-in 일반성(E4) 공정성:** 제2백본 plug-in 시 추가 파라미터/연산 증가분을 통제 안 하면 "구조 우월"이 아니라 "용량 증가" 효과. Section 4.7.

**확인된 강점(유지·강화 권장):**
- 원리 중심 reframing이 HAWK 파생물 서술을 효과적으로 제거. C1-C5 구조 명확.
- 반증가능(falsifiable) 가설 H1-H6 명시 — 탑티어 실험설계로 적절.
- 진단코드가 데이터 없이 합성검증 가능 — 재현성 강점.
