# 5. Experiment Plan (실험 계획)

본 문서는 CERBERUS 논문의 실험 섹션을 위한 실험 설계 및 실행 계획을 기술한다. 실험은 크게 **정량적 평가**, **정성적 평가**, **절제 연구(Ablation Study)**, **분석 실험**으로 구성된다.

---

## 5.1 학습 및 평가 프로토콜 (Training & Evaluation Protocol)

### 학습 파이프라인

원본 HAWK와 동일한 2-stage 학습 파이프라인을 따른다:

- **Stage 1 (사전학습)**: WebVid-2M 데이터셋에서 일반적 비디오 이해 능력 학습
  - 5개 loss 모두 활성 (L_VL, L_MV, L_ML, L_BL, L_MB)
  - 3개 브랜치 각각 개별 텍스트 감독
  - 설정: 4x A6000, batch_size=1, lr=1e-5, 160 epochs

- **Stage 2 (미세조정)**: HAWK 이상 탐지 데이터셋에서 anomaly 특화 학습
  - 3개 브랜치 임베딩 concat → 통합 language loss
  - Middle representation losses (L_MV, L_MB)만 브랜치 분리
  - 설정: Stage 1 체크포인트 로드, 동일 학습 스케줄

- **Stage 3 (평가)**: 테스트 셋에서 독립 평가
  - Video Description Generation과 Video Question-Answering을 별도 평가

### 데이터 분할

원본과 동일: 7개 데이터셋 합산 약 8,000개 비디오, 90% 학습 / 10% 테스트

### 평가 지표

| 카테고리 | 지표 | 설명 |
|---------|------|------|
| Text-Level [27] | BLEU-1, BLEU-2, BLEU-3, BLEU-4 | 생성 텍스트와 GT 간 n-gram 중복도 |
| GPT-Guided [18] | Reasonability | 논리적 추론 및 일관성 (0~1) |
| GPT-Guided [18] | Detail | 세부 정보의 수준 및 구체성 (0~1) |
| GPT-Guided [18] | Consistency | 생성 언어의 일관성 및 coherence (0~1) |

---

## 5.2 핵심 실험 (Core Experiments)

### 실험 A: 베이스라인 비교 (Main Comparison)

원본 Table 1과 동일한 형식으로 비교. 두 태스크 모두 수행:
- (A) Anomaly Video Description Generation
- (B) Anomaly Video Question-Answering

**베이스라인:**

| 모델 | 유형 | 비고 |
|------|------|------|
| Video-ChatGPT [26] | 일반 비디오 이해 | 기존 베이스라인 |
| VideoChat [15] | 일반 비디오 이해 | 기존 베이스라인 |
| Video-LLaMA [46] | 일반 비디오 이해 | 기존 베이스라인 |
| LLaMA-Adapter [47] | 일반 비디오 이해 | 기존 베이스라인 |
| Video-LLaVA [17] | 일반 비디오 이해 | 기존 베이스라인 |
| **HAWK (원본)** | VAD 특화 Dual-Branch | **핵심 비교 대상** |
| **CERBERUS (제안)** | VAD 특화 Tri-Branch | 본 논문 |

**기대 결과 테이블 형식:**

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Reasonability | Detail | Consistency |
|--------|--------|--------|--------|--------|---------------|--------|-------------|
| HAWK (원본) | 0.270 | 0.139 | 0.074 | 0.043 | 0.283 | 0.320 | 0.218 |
| **CERBERUS** | - | - | - | - | - | - | - |

### 실험 B: 데이터셋별 성능 분석 (Per-Dataset Breakdown)

7개 데이터셋 각각에 대해 HAWK vs CERBERUS 비교:

| 데이터셋 | 도메인 | Background 기여 가설 |
|---------|--------|---------------------|
| UCF-Crime [30] | 범죄 장면 | 중간 (환경 맥락) |
| ShanghaiTech [19] | 캠퍼스 | 높음 (장소 맥락) |
| CUHK Avenue [20] | 캠퍼스 | 높음 (장소 맥락) |
| UCSD Ped1 [5] | 보행자 도로 | 높음 (도로 맥락) |
| UCSD Ped2 [34] | 보행자 도로 | 높음 (도로 맥락) |
| DoTA [42] | 교통 상황 | 매우 높음 (도로 상태, 신호등) |
| UBnormal [2] | 인간 행동 | 중간 (가상 장면) |

**가설**: Background 브랜치는 장면 맥락이 이상 판별에 중요한 데이터셋(DoTA, UCSD, ShanghaiTech)에서 가장 큰 성능 향상을 보일 것이다.

---

## 5.3 절제 연구 (Ablation Study)

### Ablation A: 컴포넌트별 기여도

| 구성 | Background Branch | L_BL | L_MB | 비고 |
|------|:-:|:-:|:-:|------|
| HAWK (원본 Dual-Branch) | - | - | - | 베이스라인 |
| + Background Branch Only | O | - | - | 배경 시각 특징만 추가 |
| + Background + L_BL | O | O | - | 배경 언어 감독 추가 |
| + Background + L_MB (buggy) | O | - | O (1-cos) | 잘못된 loss 방향 |
| + Background + L_MB (correct) | O | - | O ((1+cos)/2) | 올바른 loss 방향 |
| **CERBERUS (Full)** | O | O | O ((1+cos)/2) | 전체 모델 |

**핵심 검증**: 각 컴포넌트가 독립적으로 기여하며, 전체 조합이 최고 성능을 달성하는지 확인.

### Ablation B: Loss 가중치 민감도 분석

t_3 (L_BL 가중치)와 t_4 (L_MB 가중치)를 변화시키며 성능 측정:

| t_3 | t_4 | BLEU-4 | Reasonability |
|-----|-----|--------|---------------|
| 0.01 | 0.1 | - | - |
| 0.05 | 0.1 | - | - |
| **0.1** | **0.1** | - | - |
| 0.2 | 0.1 | - | - |
| 0.1 | 0.01 | - | - |
| 0.1 | 0.05 | - | - |
| 0.1 | 0.2 | - | - |
| 0.1 | 0.5 | - | - |

### Ablation C: Loss 방향 수정 효과 (Bug Fix Impact)

동일 아키텍처에서 loss 수식만 변경하여 직접 비교:

| 구성 | L_MB 수식 | cos_sim 방향 | BLEU-4 | Reasonability |
|------|----------|-------------|--------|---------------|
| Buggy | 1 - cos_sim | 유사성 강제 (→ motion≈bg) | - | - |
| **Fixed** | (1 + cos_sim) / 2 | 비유사성 강제 (→ motion≠bg) | - | - |

**추가 측정**: 학습 수렴 후 motion-background 중간 표현 간 cosine similarity 분포 측정
- Buggy: cos_sim → +1 (높은 유사도) 으로 수렴 예상
- Fixed: cos_sim → -1 (높은 비유사도) 으로 수렴 예상

---

## 5.4 분석 실험 (Analysis Experiments)

### Analysis A: 표현 공간 시각화

- **방법**: t-SNE 또는 UMAP으로 중간 표현(X_v^c, X_m^c, X_b^c) 시각화
- **대상**: 테스트 셋에서 무작위 샘플링한 비디오들
- **비교**:
  - Dual-Branch(HAWK): appearance와 motion 클러스터만 존재
  - Tri-Branch(CERBERUS, buggy loss): motion과 background가 겹침
  - Tri-Branch(CERBERUS, correct loss): motion과 background가 분리됨
- **기대 결과**: 올바른 L_MB 적용 시 motion/background 클러스터가 명확히 분리

### Analysis B: Attention Map 시각화

- 각 브랜치의 Q-Former attention map 추출
- **기대 결과**:
  - Appearance 브랜치: 전체 프레임에 고르게 attention
  - Motion 브랜치: 움직이는 객체에 집중
  - Background 브랜치: 정적 장면(도로, 건물, 배경)에 집중
- 다양한 이상 유형에서 대표 비디오 선택 (교통사고, 보행자 이상, 범죄 등)

### Analysis C: Optical Flow 임계값 민감도

`mag_threshold`를 변화시키며 분석:

| 임계값 | Motion 영역 비율 | Background 영역 비율 | BLEU-4 | Reasonability |
|--------|:--:|:--:|--------|---------------|
| 0.05 | ~높음 | ~낮음 | - | - |
| 0.10 | - | - | - | - |
| 0.15 | - | - | - | - |
| **0.20** | - | - | - | - |
| 0.25 | - | - | - | - |
| 0.30 | - | - | - | - |
| 0.50 | ~낮음 | ~높음 | - | - |

**가설**: 임계값이 너무 낮으면 거의 모든 영역이 motion으로 분류되어 background 정보가 부족하고, 너무 높으면 대부분이 background로 분류되어 motion 정보가 부족. 0.2가 적절한 균형점.

### Analysis D: 효율성 벤치마크

| 구성 | OF 계산 시간 | 비디오 I/O 시간 | 총 전처리 시간 | Epoch당 학습 시간 |
|------|:-:|:-:|:-:|:-:|
| HAWK (Dual-Branch, 원본) | X | Y | Z | W |
| CERBERUS (Tri-Branch, 분리 처리) | ~2X | ~1.5Y | Z' | W' |
| CERBERUS (Tri-Branch, 통합 처리) | ~X | ~Y | Z'' | W'' |

**측정 방법**: 100개 비디오 샘플에 대한 평균 처리 시간 측정

---

## 5.5 정성적 평가 (Qualitative Evaluation)

### Qualitative A: Description 비교 테이블

원본 Table 2와 동일한 형식. 대표 비디오 4~6개 선택:

1. **배경 맥락이 핵심인 이상**: 도로 상태로 인한 사고 (DoTA)
2. **모션이 핵심인 이상**: 격투/폭력 장면 (UCF-Crime)
3. **양쪽 모두 중요한 이상**: 눈 덮인 도로에서 차량 미끄러짐
4. **정지 상태의 이상**: 보행자 도로 위 차량 진입 (UCSD)
5. **정상 비디오**: 오탐(false positive) 확인

각 비디오에 대해: Ground Truth, HAWK 출력, **CERBERUS 출력**을 비교하고, 차이점을 색상으로 표시 (초록=GT와 일치, 빨강=불일치)

### Qualitative B: 브랜치별 출력 시각화

선택된 비디오에 대해:
- 원본 프레임 / Motion 프레임 / Background 프레임 (나란히 배치)
- 각 브랜치 단독 입력 시 LLaMA 생성 결과 (개별 브랜치의 역할 이해)
- Tri-Branch 결합 시 최종 생성 결과

### Qualitative C: Open-World 데모

학습 데이터에 포함되지 않은 비디오에서 테스트:
- XD-Violence 데이터셋 (원본 HAWK 데모에서 사용)
- CERBERUS 가 더 맥락 인지적(context-aware)인 설명을 생성하는지 확인

---

## 5.6 실험 실행 우선순위 (Execution Priority)

**빠른 검증을 위한 권장 실행 순서:**

1. **Ablation C (Loss 방향 수정)** - 최소 compute 추가, 명확한 개선 기대
   - 동일 아키텍처에서 loss 수식 한 줄만 변경
   - Stage 2만 재학습 (~80h)

2. **Stage 2만으로 초기 검증** - 기존 Stage 1 체크포인트 활용
   - Stage 1 사전학습 건너뛰고 기존 checkpoint_127.pth 사용
   - Background 브랜치가 finetuning만으로도 효과가 있는지 검증

3. **Full 2-Stage 학습** - 초기 검증에서 개선이 확인되면 진행
   - Stage 1: ~120h+ (Tri-Branch로 인해 메모리/시간 증가)
   - Stage 2: ~80h

4. **Ablation A + B** - 컴포넌트별 기여도 확인

5. **Analysis + Qualitative** - 논문 완성을 위한 시각화

---

## 5.7 하드웨어 요구사항

| 항목 | HAWK (원본) | CERBERUS |
|------|-----------|---------|
| GPU | 4x A6000 (48GB) | 4x A6000 (48GB) |
| ViT 인코더 수 | 2 (외형 + 모션) | 3 (외형 + 모션 + 배경) |
| 예상 VRAM 증가 | - | ~30-50% (추가 ViT + Q-Former) |
| Stage 1 예상 시간 | ~120h | ~150-180h |
| Stage 2 예상 시간 | ~80h | ~100-120h |

**참고**: batch_size=1로 이미 설정되어 있어, VRAM이 부족할 경우 gradient accumulation 조정 또는 mixed precision 최적화 필요.
