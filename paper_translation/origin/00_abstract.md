# HAWK: Learning to Understand Open-World Video Anomalies

## 저자

Jiaqi Tang, Hao Lu, Ruizheng Wu, Xiaogang Xu, Ke Ma, Cheng Fang, Bin Guo, Jiangbo Lu, Qifeng Chen, Ying-Cong Chen

- The Hong Kong University of Science and Technology (Guangzhou)
- The Hong Kong University of Science and Technology
- HKUST(GZ) - SmartMore Joint Lab
- SmartMore Corporation
- The Chinese University of Hong Kong
- Zhejiang University
- Northwestern Polytechnical University

---

## Abstract (초록)

비디오 이상 탐지(Video Anomaly Detection, VAD) 시스템은 자율적으로 교란을 모니터링하고 식별할 수 있어, 수작업과 관련 비용의 필요성을 줄여준다. 그러나 현재의 VAD 시스템은 장면에 대한 피상적인 의미 이해와 최소한의 사용자 상호작용으로 인해 제한되는 경우가 많다. 또한, 기존 데이터셋의 만연한 데이터 부족은 개방 세계(open-world) 시나리오에서의 적용 가능성을 제한한다.

본 논문에서는 대화형 대규모 시각-언어 모델(Visual Language Model, VLM)을 활용하여 비디오 이상을 정밀하게 해석하는 새로운 프레임워크인 **HAWK**를 소개한다. 비정상 비디오와 정상 비디오 간의 모션 정보 차이를 인식하여, HAWK는 이상 식별을 향상시키기 위해 모션 모달리티를 명시적으로 통합한다.

모션 주의(motion attention)를 강화하기 위해, 모션 및 비디오 공간 내에서 보조 일관성 손실(auxiliary consistency loss)을 구성하여 비디오 브랜치가 모션 모달리티에 집중하도록 유도한다. 더 나아가, 모션에서 언어로의 해석을 개선하기 위해 모션과 그 언어적 표현 간에 명확한 감독 관계를 수립한다.

또한, 8,000개 이상의 이상 비디오에 언어 설명을 주석 처리하여 다양한 개방 세계 시나리오에서 효과적인 학습을 가능하게 하였으며, 사용자의 개방형 질문을 위한 8,000개의 질문-답변 쌍도 생성하였다.

최종 결과는 **HAWK**가 비디오 설명 생성과 질문 답변 모두에서 기존 베이스라인을 능가하며 SOTA 성능을 달성함을 보여준다. 코드/데이터셋/데모는 https://github.com/jqtangust/hawk 에서 공개될 예정이다.
