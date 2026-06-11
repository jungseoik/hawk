# 4. Experiments (실험)

본 절에서는 제안하는 상보적 시각 분해(Complementary Visual Decomposition, CVD) 원리의 타당성을 검증하기 위한 실험 프로토콜과 평가 설계를 기술한다. 본 연구의 핵심 주장은 **개별 성능 수치의 우월성**에 머무르지 않고, *분해가 진정으로 상보적인가*, *정적 스트림이 이상 판별에 인과적으로 기여하는가*, *방향성 목적함수가 표현 분리를 야기하는가*, *원리가 특정 백본에 종속되지 않고 일반화되는가* 라는 네 가지 검증 가능한(falsifiable) 질문으로 구체화된다. 이를 위해 표준 정량 비교(Section 4.3)에 더하여, 본 연구가 새로 도입하는 **진단 평가 스위트**(Section 4.2)를 중심으로 네 개의 진단 실험(E1–E4, Section 4.4–4.7)을 설계한다.

<!-- [Table 위치] 모든 정량 결과 표는 학습 완료 후 채워진다. 본 절은 프로토콜·가설·평가 지표를 확정한다. -->

---

## 4.1 실험 프로토콜 (Experimental Protocol)

### 학습 단계 (Training Stages)

제안 프레임워크는 2단계로 학습된다.

- **Stage 1 (사전 학습, Pretraining):** 대규모 비디오-캡션 코퍼스(WebVid)에서 세 스트림(Appearance, Dynamic, Static)을 각각 독립적인 텍스트 감독으로 학습한다. 이 단계에서 다섯 개의 손실 항(L_VL, L_sim, L_ML, L_BL, L_dis)이 모두 활성화되며(Section 3, 수식 10), 각 스트림은 자신의 모달리티에 특화된 표현을 획득한다.
- **Stage 2 (미세 조정, Finetuning):** 이상 이해(anomaly understanding) 데이터셋에서 세 스트림의 임베딩을 연결(concatenation)한 후 통합 언어 손실로 미세 조정한다. 중간 표현 손실(L_sim, L_dis)은 스트림 분리를 유지하기 위해 개별 압축 표현에 대해 독립적으로 계산된다. 이 단계에서 시각 토큰 길이는 `num_video_query_token × 3`으로 확장된다.

### 검증 무대 (Validation Testbed)

CVD 원리는 일반 비디오-언어 이해에 적용 가능하나, 그 효용을 가장 측정 가능하고 반증 가능하게 드러내는 무대로 **비디오 이상 이해(Video Anomaly Understanding)** 를 채택한다. 상당 비율의 이상은 움직임 자체가 아니라 *어디서·어떤 조건에서* 발생하는가, 즉 정적 장면 맥락에 의해 그 위험성이 규정되며(예: 고속도로 차로 위 정지 보행자, 결빙 노면에서의 미끄러짐), 이 비율은 본 연구의 Background-critical 큐레이션(Section 4.2)에서 도메인별로 정량화한다. 따라서 정적 스트림의 기여는 이 설정에서 가장 명확히 분리·정량화된다. 평가에는 비디오 설명 생성(description generation)과 질의 응답(question-answering)의 두 과제를 사용한다.

### 평가 지표 (Evaluation Metrics)

| 범주 (Category) | 지표 (Metric) | 설명 |
|---|---|---|
| Text-Level | BLEU-1 ~ BLEU-4 | 생성 텍스트와 정답 간 n-gram 중복도 |
| GPT-Guided | Reasonability / Detail / Consistency | 논리성·구체성·일관성 (0–1) |
| **Diagnostic (신규)** | **CDS** (Complementary Disentanglement Score) | 스트림 간 비중복성과 합집합 정보 보존의 결합 (Section 4.2, 수식 11–13) |
| **Diagnostic (신규)** | **BSI** (Background Sensitivity Index) | 정적 맥락 활용의 인과적 민감도 (Section 4.2, 수식 14) |

### 베이스라인 (Baselines)

일반 비디오-언어 모델(Video-ChatGPT, VideoChat, Video-LLaMA, LLaMA-Adapter, Video-LLaVA)과, **정적 스트림을 두지 않는 이중 스트림(dynamic-only) 접근**을 핵심 비교 대상으로 둔다. 후자는 CVD에서 정적 스트림을 제거한 특수 사례(이 경우 동적-정적 손실 L_dis는 자동으로 소멸)에 해당하므로, 정적 스트림의 순수 기여를 분리해 측정하는 통제된 베이스라인 역할을 한다.

---

## 4.2 진단 평가 스위트 (Diagnostic Evaluation Suite)

표준 텍스트 지표는 분해가 *상보적으로 작동하는지*를 직접 측정하지 못한다. 본 연구는 분해의 품질을 측정하는 두 가지 진단 지표를 새로 도입한다(기여 C5).

### 상보적 분리도 점수 (Complementary Disentanglement Score, CDS)

세 스트림의 중간 압축 표현 `z_a`(외형), `z_m`(동적), `z_b`(정적)에 대해, 동적·정적 표현이 *서로 중복되지 않으면서도*(non-redundant) 두 표현의 합집합이 *전체 프레임 정보를 보존*(coverage)하는 정도를 단일 스칼라로 정량화한다. 표현 간 정렬도는 중심화 커널 정렬(Centered Kernel Alignment, CKA)로 측정한다.

> Coverage = CKA([z_m ; z_b], z_a) --- (11)
>
> Redundancy = CKA(z_m, z_b) --- (12)
>
> CDS = Coverage × (1 − Redundancy) --- (13)

여기서 `[z_m ; z_b]`는 두 표현의 연결이며, CKA(·,·) ∈ [0,1]은 표현 정렬도이다. CDS ∈ [0,1]는 (i) 동적·정적 표현이 비중복적이고(낮은 Redundancy), (ii) 그 합집합이 외형 스트림이 담은 전체 프레임 정보를 잘 설명할 때(높은 Coverage) 최대가 된다. CKA 외에 가우시안 근사 상호정보량 `I(z_m; z_b)`와 KSG(Kraskov) k-최근접 이웃 추정량을 보조 지표로 함께 보고하여, 정보이론적 비중복성(Section 3.1.3)을 직접 검증한다.

### 배경 민감도 지표 (Background Sensitivity Index, BSI)

모델이 정적 맥락을 *인과적으로* 활용하는지를 측정하기 위해, 동일한 전경 움직임을 상이한 배경에 합성하는 반사실적(counterfactual) 프로브를 정의한다(Section 4.5). 동일 전경 `fg`를 안전 배경과 위험 배경에 각각 합성했을 때 모델 응답의 변화량을 다음과 같이 측정한다.

> BSI = 1 − sim( R(fg ⊕ bg_safe), R(fg ⊕ bg_danger) ) --- (14)

여기서 `R(·)`은 모델의 생성 응답, `⊕`는 상보 분해 마스크를 사용한 합성 연산자(`M⊙fg + (1−M)⊙bg`), `sim(·,·)`은 응답 간 유사도이다(공개 구현은 의존성 없는 토큰 단위 Jaccard를 기본으로 제공하며, 문장 인코더 주입 시 임베딩 코사인으로 대체된다 — 본 실험은 문장 임베딩 코사인을 사용한다). 정적 스트림을 활용하는 모델은 배경 교체에 따라 응답이 변하므로 BSI가 높다.

**Confound 통제 — 증분 민감도.** 외형 스트림은 원본 프레임을 입력받으므로 정적 스트림이 없는 모델도 배경 정보를 일부 흡수하여 BSI > 0을 보일 수 있다. 따라서 정적 *전용* 스트림의 순수 기여를 분리하기 위해, 절대 BSI가 아니라 **증분 배경 민감도** `ΔBSI = BSI(전체 모델) − BSI(정적 스트림 제거 모델)`를 주 지표로 보고한다. `ΔBSI > 0`은 정적 스트림이 외형 스트림이 흡수하지 못한 배경 단서를 추가로 활용함을 의미한다.

### Background-critical 진단 벤치마크

이상 판별이 정적 장면 맥락에 의해 결정되는 클립만을 선별한 진단용 평가 집합을 구성한다. 각 클립은 `motion_sufficient`(움직임만으로 판별 가능), `context_dependent`(맥락 필요), `context_critical`(움직임은 정상/부재, 배경만이 이상을 드러냄)으로 주석되며, 벤치마크는 후자의 두 범주로 구성된다. 추가로, 반사실적 배경 교체(BSI)를 통한 통제 실험을 병행한다. 데이터 큐레이션·주석 스키마와 매니페스트 형식은 부록 및 공개 코드의 `experiments/bg_critical_benchmark/`에 정의한다.

---

## 4.3 핵심 비교 (Main Comparison)

이상 비디오 설명 생성(A)과 질의 응답(B)에 대해 베이스라인과 비교한다.

**Table 1. 정량적 성능 비교 (placeholder — 학습 후 기입).**

| Method | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | Reasonability | Detail | Consistency |
|---|---|---|---|---|---|---|---|
| Video-ChatGPT | - | - | - | - | - | - | - |
| VideoChat | - | - | - | - | - | - | - |
| Video-LLaMA | - | - | - | - | - | - | - |
| LLaMA-Adapter | - | - | - | - | - | - | - |
| Video-LLaVA | - | - | - | - | - | - | - |
| Dynamic-only (정적 스트림 제거) | - | - | - | - | - | - | - |
| **CERBERUS (제안)** | - | - | - | - | - | - | - |

### 데이터셋별 분석 (Per-Dataset Breakdown)

데이터셋별로 정적 맥락의 중요도가 상이하다는 가설을 검증한다.

| 데이터셋 | 도메인 | 정적 맥락 기여 가설 |
|---|---|---|
| 교통 상황 (예: DoTA) | 도로 | 매우 높음 (노면 상태·신호) |
| 보행자 도로 (예: UCSD) | 도로 | 높음 |
| 캠퍼스 (예: ShanghaiTech, Avenue) | 장소 | 높음 |
| 범죄 장면 (예: UCF-Crime) | 환경 | 중간 |
| 인간 행동 (예: UBnormal) | 가상 장면 | 중간 |

> **가설 H1.** 정적 스트림의 성능 향상은 장면 맥락이 이상 판별에 결정적인 도메인(교통·도로·장소)에서 가장 크게 나타난다.

---

## 4.4 E1: 상보적 분리도 분석 (Disentanglement Analysis)

**목표.** 동적·정적 표현이 비중복적이면서 합집합이 전체 정보를 보존함을 정량적으로 입증한다(기여 C2). 테스트 분할에서 추출한 중간 표현(`z_a, z_m, z_b`)에 대해 CDS(수식 13), CKA, 상호정보량, 코사인 유사도 분포를 측정한다.

> **가설 H2.** 올바른 방향성 손실(L_dis = (1+cos)/2)을 적용한 전체 모델은 (i) `I(z_m; z_b)`와 CKA(z_m, z_b)가 낮고, (ii) CDS가 최고값을 가진다. L_dis를 제거하면 비중복성이 약화되어 CDS가 감소하고, 손실 방향을 뒤집으면(1−cos) 표현이 붕괴하여 CDS가 최소가 된다.

**Table 2. 분리도 진단 (placeholder).**

| 구성 | CKA(z_m,z_b) ↓ | I(z_m;z_b) ↓ | Coverage ↑ | Probing Acc (z_m / z_b) ↑ | **CDS** ↑ |
|---|:-:|:-:|:-:|:-:|:-:|
| 동일 입력 통제군 (상한) | 높음(기대) | - | - | - | 낮음(기대) |
| L_dis 제거 | - | - | - | - | - |
| L_dis = 1 − cos (역방향) | - | - | - | - | - |
| **L_dis = (1 + cos)/2 (제안)** | - | - | - | - | - |

(Probing Acc 열은 CDS confound 방어를 위한 동반 지표로, 비중복성과 과제 유용성을 동시에 확인한다.)

**Confound 통제.** CDS는 두 가지 자명한 방식으로 높아질 수 있어 단독으로는 충분치 않다. 첫째, `z_a`가 전체 프레임을 보고 `z_m`·`z_b`가 그 부분을 보므로 Coverage는 분리가 약해도 높을 수 있다. 둘째, 두 스트림이 *과제와 무관한 노이즈*만 학습해도 Redundancy는 낮아진다. 이를 방어하기 위해 (i) CDS와 함께 각 스트림의 **선형 프로빙 정확도**(linear probing accuracy, 과제 정보 `I(z; y)`의 대리 지표)를 반드시 동반 보고하여 "비중복이면서 동시에 두 스트림 모두 과제에 유용함"을 입증하고, (ii) 두 스트림에 동일 입력을 주어 Redundancy가 높게 나오도록 강제한 **상한 통제군**과, L_dis를 제거한 통제군을 비교하여 CDS가 의도대로 변별함을 보인다. 진단 지표 자체의 타당성은 데이터·학습 없이 합성 표현에 대해 검증하였으며(`experiments/disentanglement.py`), 상보·부분중복·붕괴 세 체제에서 CDS가 단조적으로 분리됨을 확인하였다(complementary > partial > collapsed; 실모델 결과는 학습 후 보고).

<!-- [Figure 위치] 중간 표현 공간의 t-SNE/UMAP 시각화: (a) 이중 스트림 (b) Tri-Stream + 역방향 손실(겹침) (c) Tri-Stream + 제안 손실(분리) -->

---

## 4.5 E2: Background-critical 진단 평가 (Background-Critical Evaluation)

**목표.** 모델이 정적 맥락을 인과적으로 활용하여 이상을 판별함을 입증한다(기여 C1, C5).

- **(a) 큐레이션 평가:** `context_dependent` 및 `context_critical` 부분집합에서의 정확도를 전체 집합과 대비하여, 정적 스트림이 맥락 의존 이상에서 더 큰 이득을 줌을 확인한다.
- **(b) 반사실적 배경 교체:** 동일 전경 움직임을 안전/위험 배경에 합성하고(수식 14, `experiments/bg_critical_benchmark/background_swap.py`), BSI로 응답 변화를 측정한다.

> **가설 H3.** 외형 스트림이 배경을 일부 흡수하므로 두 모델 모두 BSI > 0일 수 있으나, 정적 *전용* 스트림을 갖춘 전체 모델은 더 높은 BSI를 보여 `ΔBSI = BSI(전체) − BSI(정적 스트림 제거) > 0`이 성립한다. 또한 정적 맥락이 결정적인 `context_critical` 부분집합에서 전체 모델의 정확도 이득이 가장 크다.

**Table 3. Background-critical 진단 (placeholder).**

| 구성 | Acc (context_critical) | Acc (전체) | BSI |
|---|:-:|:-:|:-:|
| Dynamic-only (정적 스트림 제거) | - | - | - |
| **CERBERUS** | - | - | - |
| **ΔBSI (= 전체 − 제거)** | | | **- (>0 기대)** |

합성 프로브 상에서 합성 연산자가 전경을 보존하고 배경만 교체함을, 그리고 BSI가 배경 둔감/배경 인지 응답을 구분함을 검증하였다(실모델 결과는 학습 후 보고).

---

## 4.6 E3: 손실 방향 인과 검증 (Loss-Direction Causal Verification)

**목표.** 비유사성 손실의 *방향*이 표현 분리를 야기하는 인과 요인임을 통제 실험으로 입증한다(기여 C2). 동일 아키텍처·동일 데이터에서 손실 수식만 교체한다.

| 구성 | L_dis 수식 | 수렴 방향 | 기대 |
|---|---|---|---|
| 역방향 | 1 − cos(z_m, z_b) | cos → +1 | 표현 붕괴(중복) |
| **제안** | (1 + cos(z_m, z_b))/2 | cos → −1 | 표현 분리(비중복) |

학습 수렴 후 `cos(z_m, z_b)` 분포와 하류 성능을 함께 보고한다. 손실 방향의 인과적 효과는 데이터·학습 없이도 통제된 합성 최적화로 즉시 입증되며(`experiments/loss_direction.py`), 역방향 손실은 `cos → +1`(붕괴), 제안 손실은 `cos → −1`(분리)로 수렴함을 확인하였다(실모델 학습 결과는 추후 보고).

> **가설 H4.** 손실 방향만 올바르게 교정하면(다른 모든 조건 동일) 분리도(E1)와 하류 이상 이해 성능이 동반 향상된다.

---

## 4.7 E4: 상보성 일반성 검증 (Complementarity Generality)

**목표.** CVD(상보 분해 + 방향성 분리 목적함수)가 특정 백본·특정 과제에 종속되지 않음을 입증하여, 기여를 일반 원리로 정립한다(기여 C1).

- **백본 일반성:** 주 백본 외 제2의 비디오-언어 백본에 CVD를 plug-in하여 일관된 향상을 확인한다.
- **과제 일반성:** 이상 이해 외 일반 비디오 캡셔닝에 적용하여 향상 여부를 확인한다.

> **가설 H5.** CVD 적용 시 성능 향상(Δ = with_CVD − without_CVD)이 백본·과제 조합 전반에서 일관되게 양(+)으로 나타난다.

**용량 confound 통제.** 정적 스트림 추가는 파라미터·연산을 증가시키므로, "구조의 효과"와 "단순 용량 증가"를 분리해야 한다. 이를 위해 동일 파라미터 예산의 통제 baseline — (i) 외형 스트림을 하나 더 추가한 모델, (ii) 단일 스트림에서 시각 토큰을 `×3`으로 늘린 모델 — 과 비교하여, CVD의 향상이 용량이 아닌 상보 분해 구조에서 비롯됨을 입증한다. 프로토콜과 구성 행렬은 `experiments/configs/plugin_ablation.yaml`에 정의한다.

---

## 4.8 절제 연구 및 분석 (Ablation & Analysis)

### 컴포넌트 절제 (Component Ablation)

정적 시각 스트림, 정적-언어 감독(L_BL), 방향성 분리 손실(L_dis)을 *누적적으로* 추가하며(각 행은 직전 행에서 단 하나의 요소만 변경) 각 요소의 독립적 기여를 분리한다(구성 행렬: `experiments/configs/ablation.yaml`). L_dis의 *방향*(역방향 vs 제안)에 대한 통제 비교는 E3(Section 4.6)에서 별도로 다룬다.

| 구성 | Static Stream | L_BL | L_dis | 변경점 |
|---|:-:|:-:|:-:|---|
| Dynamic-only | - | - | - | 베이스라인(특수 사례) |
| + Static Stream | O | - | - | 정적 시각 스트림 추가 |
| + L_BL | O | O | - | 정적-언어 감독 추가 |
| **CERBERUS (Full)** | O | O | (1+cos)/2 | 방향성 분리 손실 추가 |

### 분할 임계값 민감도 (Partition Threshold Sensitivity)

옵티컬 플로우 크기 임계값 τ를 변화시키며 동적/정적 영역 비율, 하류 성능, CDS를 측정한다(`experiments/configs/threshold_sweep.yaml`).

> **가설 H6.** τ가 지나치게 작으면 대부분이 동적으로 분류되어 정적 스트림이 빈약해지고, 지나치게 크면 그 반대가 되어, τ ≈ 0.2 부근에서 균형점이 형성된다.

### 효율성 (Efficiency)

통합 플로우 연산(플로우 1회 → 두 마스크)이 분리 연산 대비 옵티컬 플로우 계산을 약 50%, 비디오 I/O를 약 33% 절감함을 측정한다(`experiments/efficiency_benchmark.py`). 본 최적화는 모델 출력에 영향을 주지 않는 순수 계산 효율 개선이다.

| 구성 | 옵티컬 플로우 | 비디오 I/O | 클립당 처리 시간 |
|---|:-:|:-:|:-:|
| 분리 연산 | O(2N) | 3회 | - |
| **통합 연산 (제안)** | O(N) | 2회 | - |

---

## 4.9 정성 평가 (Qualitative Evaluation)

대표 비디오에 대해 정답, Dynamic-only 출력, CERBERUS 출력을 비교한다. 선정 기준은 (1) 배경 맥락이 핵심인 이상, (2) 움직임이 핵심인 이상, (3) 양쪽 모두 중요한 이상, (4) 정지 상태의 이상, (5) 정상 비디오(오탐 점검)이다. 아울러 각 스트림의 입력(원본/동적/정적 프레임)과 스트림별 단독 생성 결과를 나란히 제시하여, 정적 스트림이 장면 서술(노면·기상·구조물 등)을 담당함을 시각적으로 확인한다. 학습 데이터에 포함되지 않은 비디오에 대한 Open-World 데모도 함께 제시한다.

<!-- [Table 위치] 정성 비교(GT / Dynamic-only / CERBERUS). 초록=정답 일치, 빨강=불일치. -->
<!-- [Figure 위치] 스트림별 입력 프레임(원본/동적/정적)과 스트림별 단독 생성 결과. -->

---

## 4.10 재현성 (Reproducibility)

모든 진단 실험은 데이터·학습 없이도 합성 입력으로 검증 가능한 형태로 공개된다(`experiments/`, 매핑은 `experiments/README.md`). 진단 지표(CDS·BSI), 손실 방향 통제 실험, 효율성 벤치마크는 즉시 재현 가능하며, 학습이 필요한 실험은 환경 구성(`scripts/setup_env.sh`)과 실행 스크립트(`scripts/run_stage{1,2}.sh`, `run_eval.sh`, `extract_representations.py`)로 제공된다.
