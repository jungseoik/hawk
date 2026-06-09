# CERBERUS: Three-Headed Video Understanding for Open-World Anomaly Detection

## 저자 (Authors)

[저자 정보 추가 예정]

---

## Abstract (초록)

비디오 이상 탐지(Video Anomaly Detection, VAD) 시스템은 자율적으로 이상을 모니터링하고 식별할 수 있으나, 현재의 접근 방식은 장면에 대한 피상적 의미 이해와 제한된 사용자 상호작용으로 인해 한계를 보인다. 최근 HAWK는 대규모 시각-언어 모델(VLM)에 모션 모달리티를 명시적으로 통합하여 이상 탐지 성능을 크게 향상시켰다. 그러나 HAWK의 Dual-Branch 아키텍처(외형 + 모션)는 동적 영역에만 집중하여, 이상의 원인을 설명하는 **정적 장면 맥락**(도로 상태, 주변 환경 등)을 간과한다.

본 논문에서는 Background 브랜치를 추가한 **Tri-Branch 아키텍처** CERBERUS를 제안한다. 그리스 신화의 머리 셋 달린 수호자 케르베로스(Cerberus)에서 착안하여, 세 개의 독립적 시각 인코더가 각각 외형(Appearance), 움직임(Motion), 배경(Background)이라는 고유한 관점에서 비디오를 감시하는 구조를 반영하였다. 핵심 아이디어는 옵티컬 플로우 기반으로 비디오를 동적 영역(Motion)과 정적 영역(Background)으로 수학적으로 분리하고, 각각에 대해 전용 인코더를 할당하는 것이다. 이때 Motion + Background = 원본 프레임이라는 **상보적 관계**가 성립하여, 정보 손실 없이 비디오의 모든 시각 정보를 두 상보적 관점에서 학습한다.

또한, 상보적 표현 학습을 위해 appearance-motion 간 **유사성 강제**와 motion-background 간 **비유사성 강제**를 수학적으로 구분한 확장된 loss 설계를 도입하여, 각 브랜치가 중복 없이 고유한 정보를 학습하도록 한다. 아울러, 단일 옵티컬 플로우 연산으로 motion과 background 마스크를 동시 생성하는 효율적 처리 방식을 통해 전처리 시간을 약 50% 절감한다.

<!-- [실험 결과 추가 예정: CERBERUS는 7개 이상 탐지 데이터셋에서 Text-Level 및 GPT-Guided 지표 모두에서 기존 HAWK를 포함한 베이스라인을 상회하는 성능을 달성한다.] -->
