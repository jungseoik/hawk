# CERBERUS: Complementary Visual Decomposition for Video-Language Understanding (상보적 시각 분해를 통한 비디오-언어 이해)

## 저자 (Authors)

[저자 정보 추가 예정]

---

## Abstract (초록)

비디오를 언어로 해석하는 모델, 예컨대 Video-LLaMA·Video-ChatGPT 등 주류 비디오-언어 모델은 프레임을 단일 토큰 시퀀스로 인코딩하며, 그 결과 의미상 구별되는 시각 요인(visual factor) — 외형·움직임·정적 장면 맥락 — 이 단일 표현 공간에서 뒤섞인다(entangle). 본 논문에서는 이러한 한계를 해소하기 위한 일반 원리로서 **상보적 시각 분해(Complementary Visual Decomposition, CVD)**를 제안한다. CVD는 옵티컬 플로우(optical flow) 크기에 기반한 이진 마스크 `M`을 이용하여 임의의 프레임 `x`를 공간적으로 서로소인 두 스트림, 즉 동적 스트림(dynamic stream) `M ⊙ x`와 정적 스트림(static stream) `(1 − M) ⊙ x`로 분할한다. 두 스트림의 화소 단위 합은 원본 프레임과 정확히 일치하며(`M ⊙ x + (1 − M) ⊙ x = x`), 모든 화소의 원본 값이 두 스트림 중 하나에 변형 없이 분배되는 **화소값 보존 상보 분할(value-preserving complementary partition)**을 이룬다. 이는 옵티컬 플로우 이미지나 RGB 차분(RGB-difference)과 같이 원본 RGB와 상이한 신호 공간에 거주하거나 화소값을 잔차로 치환하며 정적 외형을 폐기하는 기존의 손실적(lossy) 모션 모델링과 근본적으로 구별된다.

기하적 상보성이 곧 표현적 분리를 보장하지는 않으므로, 본 논문은 **방향성 분리 목적함수(directional disentanglement objective)**를 도입한다. 외형-동적 스트림 간에는 유사성을, 동적-정적 스트림 간에는 비유사성을 강제하는 비대칭(asymmetric) 손실 설계를 통해, 기하적 상보성을 표현 수준의 비중복성으로 전환한다. 이는 스트림 간 상호정보량 `I(z_m; z_b)`를 최소화하면서 각 스트림의 과제 정보 `I(z; y)`를 보존하는 최소 충분(minimal-sufficient) 표현으로 모델을 유도한다. 나아가 캡션을 의존 구문 분석하여 각 시각 스트림에 정합되는 **모달리티 정합 언어 감독(modality-matched linguistic supervision)**을 부여함으로써, 시각 분해를 언어 공간에 정초(ground)한다.

본 원리의 효용은 가장 측정 가능하고 반증 가능한 검증 무대(testbed)인 비디오 이상 탐지(Video Anomaly Detection, VAD)에서 외형/동적/정적 세 스트림 인스턴스로 구현하여 검증한다. VAD에서 이상의 원인은 흔히 정적 장면 맥락에 있으나, 정적 스트림을 두지 않는 기존 분해 방법은 이를 구조적으로 누락한다. 본 연구의 3-스트림 모델 CERBERUS는 머리 셋 달린 수호자에 빗대어, 각 스트림이 고유한 관점에서 비디오를 감시하는 CVD 원리의 인스턴스화이다. 우리는 표준 정량 비교에 더하여, 분해의 상보성을 직접 측정하는 진단 스위트(분리도 지표 CDS, 배경 민감도 지표 BSI)와 네 가지 반증 가능한 진단 실험으로 제안 구조의 우월성을 검증한다.

<!-- 정량 결과 추가 예정 (학습 완료 후) -->
