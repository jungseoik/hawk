# CERBERUS — Research Reframing & Experiment Master Plan

> 📌 본 문서는 **신규 연구 CERBERUS**의 권위 있는 설계·집필·실험 마스터 플랜입니다.
> 베이스 HAWK(Dual-Branch)와의 구분은 [`../CLAUDE.md`](../CLAUDE.md)를 참고하세요.
> 이 문서는 논문 집필 에이전트(`academic-paper-writer`)와 검증 에이전트(`toptier-paper-reviewer`)의
> 1차 브리프이자, 실험 코드(`../experiments/`)의 설계 근거입니다.

---

## 0. 동기: 무엇을 바꾸는가 (Why this rewrite)

기존 초안은 **"HAWK의 Dual-Branch에 Background 브랜치를 더해 Tri-Branch로 만들었다"** 는
HAWK 파생물(derivative) 서술이 지배적이다. 이는 (a) 노벨티를 "한 브랜치 추가"로 왜소화하고,
(b) 기여를 특정 베이스라인에 종속시키며, (c) 일반 원리로서의 가치를 가린다.

**재프레이밍 원칙(사용자 결정): "원리 중심 + VAD는 검증 무대".**
핵심 아이디어 — *옵티컬 플로우 역마스크 기반 상보적 시각 분해* — 를 **일반 원리**로 격상하고,
HAWK는 이 원리가 부분적으로만 구현된 **특수 사례 / 베이스라인 중 하나**로 위치시킨다.
VAD는 이 원리의 효용이 가장 측정 가능하고 반증 가능한 **검증 무대(testbed)** 로 둔다.

---

## 1. 핵심 원리 (The Principle): Complementary Visual Decomposition (CVD)

### P1. 무손실 상보 분할 (Lossless complementary partition)

임의의 프레임 `x`를 옵티컬 플로우 크기에 따른 이진 마스크 `M`으로 **상호배타적·전체포괄적**
두 스트림으로 분할한다.

> `D(x) = ( M ⊙ x , (1 − M) ⊙ x )`  s.t.  `M ⊙ x + (1 − M) ⊙ x = x`

- **동적 스트림(dynamic)** = `M ⊙ x` (움직이는 영역)
- **정적 스트림(static / residual)** = `(1 − M) ⊙ x` (정적 장면 맥락)

**선행 분해와의 차별점(핵심 노벨티):** 기존 two-stream / motion modeling은
옵티컬 플로우 이미지·RGB-difference 같은 **변환된 별도 모달리티**를 파생하여
원본 RGB와 *다른 공간*에 살며 정적 외형을 버린다(lossy, 공간 불일치).
CVD는 **원본 픽셀 공간의 정확한 분할**이다 →
(i) 합집합의 정보 무손실, (ii) 입력 수준에서 픽셀 중첩 0(구조적 비중복).
즉 "모션 모델링"을 **"상보적 장면 인수분해(scene factorization)"** 로 일반화한다.

### P2. 기하적 상보성 → 표현적 분리 (Geometric → representational disentanglement)

입력이 상보적이라고 표현이 분리되는 것은 아니다(두 인코더가 중복 특징으로 붕괴 가능).
**방향성 목적함수(directional objective)** 로 표현 수준 분리를 강제한다.

- **유사성 강제** (appearance ↔ dynamic): `L_sim = 1 − cos(z_a, z_m)` → 동적 영역이 외형의 부분집합이므로 상호 강화.
- **비유사성 강제** (dynamic ↔ static): `L_dis = (1 + cos(z_m, z_b)) / 2` → 상보적 비중첩이므로 표현 비중복.

두 손실의 **비대칭 형태**가 형식적 메커니즘이다.
**정보이론적 해석:** 중복성 `I(z_m; z_b)`는 낮추고, 각 스트림의 과제 정보 `I(z; y)`는 보존 →
*최소 충분(minimal-sufficient) · 인수분해된* 표현으로 유도. (E1에서 측정, E3에서 인과 검증.)

### P3. 모달리티 정합 언어 감독 (Modality-matched linguistic supervision)

분할을 언어 공간에서도 거울처럼 반영한다. 캡션을 의존 구문 분석하여
- **행동 언어(action-language)** = 동사 + 주어/목적어 개체 → 동적 스트림 감독
- **장면 언어(scene-language)** = 비-논항 명사 + 형용사 → 정적 스트림 감독
각 시각 스트림이 정합된 텍스트 감독을 받아 분리를 언어 공간에 정초(ground)한다.

### P4. 효율적 통합 연산 (Efficiency)

플로우 1회 계산 → 두 마스크 동시 산출(`bg = 1 − motion`). 실용적·부수적 기여.

---

## 2. 기여 재정의 (Contributions, principle-first)

- **C1 (원리):** CVD — 비디오-언어 모델을 위한 무손실 상보 픽셀 공간 분할. HAWK의 Dual-Branch는 정적 스트림이 없는 특수 사례.
- **C2 (목적함수):** 기하적 상보성을 표현적 비중복으로 전환하는 방향성 분리 목적함수 + 정보이론적 정당화.
- **C3 (언어):** 상보적·모달리티 정합 언어 감독.
- **C4 (효율):** 통합 플로우 연산.
- **C5 (진단 도구):** 분해가 *진짜로 상보적인지* 측정하는 진단 스위트 — 정보이론적 분리도 지표 **CDS** + **Background-critical 벤치마크 프로토콜**. 본 연구를 넘어 재사용 가능.

> CERBERUS = 케르베로스(머리 셋). Appearance / Dynamic / Static 세 수호자가 각자의 관점으로 비디오를 감시 — CVD 원리의 3-스트림 인스턴스화.

---

## 3. 실험 설계 (Experiments) — 결과 없이 구조 우수성 증명

타깃: **Top-tier (CVPR/NeurIPS/ICCV)**. 결과 수치는 미기재(placeholder), 설계·프로토콜·가설을 엄밀히 제시.
4가지 노벨티 증명 전략 전부 채택.

### E1 — 정보이론적 분리도 측정 (Disentanglement Measurement) → C2, C5
- **대상:** 중간 표현 `z_a, z_m, z_b` (모델 `middle_result*`에서 추출).
- **지표:** (a) CKA(linear+RBF), (b) MI 추정(Kraskov kNN / MINE), (c) cosine-sim 분포, (d) **신규 CDS**.
- **CDS 정의:** 비중복성(낮은 `I(z_m;z_b)`)과 과제 정보 합집합 보존을 결합한 단일 스칼라.
- **가설표:** fixed L_dis → `I(z_m;z_b)`·CKA 낮음 / L_dis 제거 → 중간 / buggy L_dis → 최고(붕괴).
- **지금 실행 가능:** 지표 코드 + 합성(Gaussian) 시연으로 CDS가 의도대로 동작함을 검증. 실모델은 추후.
- 코드: `experiments/disentanglement.py`

### E2 — Background-critical 진단 벤치마크 → C1, C5
- **프로토콜:** 모션은 고정하고 배경에 따라 이상 판정이 뒤집히는 케이스 집합 구성.
  - (a) DoTA/UCF/ShanghaiTech에서 'motion-insufficient' 주석 스키마로 큐레이션.
  - (b) **반사실적 배경 교체 프로브(counterfactual background swap):** 동일 전경 모션을 안전/위험 배경에 합성 → 모델 설명·이상판정이 적절히 바뀌는지 측정(배경 활용의 인과 검증).
- **신규 지표 BSI(Background Sensitivity Index):** 배경 교체 시 응답 변화량.
- **지금:** 매니페스트 스키마·합성 컴포지터 스텁. 데이터는 추후.
- 코드: `experiments/bg_critical_benchmark/`

### E3 — Loss 방향 인과 검증 (Loss-direction Causal Verification) → C2
- **controlled 비교:** buggy(`1−cos`) vs fixed(`(1+cos)/2`), 그 외 동일.
- **측정:** `cos(z_m,z_b)` 수렴 궤적(+1 vs −1), 하류 지표 차, t-SNE/UMAP 클러스터 분리.
- **지금:** 두 학습 벡터 합성 시연으로 각 손실의 수렴 방향을 즉시 입증. 실학습은 추후.
- 코드: `experiments/loss_direction.py`, `experiments/representation_viz.py`

### E4 — 상보성 일반성 검증 (Complementarity Generality) → C1
- CVD(분해 + 분리 목적함수)를 **다른 backbone / 다른 과제(일반 비디오 캡셔닝)** 에 plug-in → 일관된 향상으로 HAWK 비종속성 입증.
- **plug-in ablation 프로토콜.** 코드: `experiments/configs/plugin_ablation.yaml` + 설계서.

### 표준 실험 (Standard)
- Main 비교(HAWK 포함 baseline), per-dataset breakdown(배경 기여 가설), 컴포넌트 ablation,
  플로우 임계값 민감도, 효율성 벤치마크(`experiments/efficiency_benchmark.py` — 지금 실행 가능),
  정성/브랜치별 출력/Open-world.

---

## 4. 실행 가능 세팅 범위 (Setup scope — 데이터셋은 맨 나중)

1. **가상환경(Blackwell 호환):** `scripts/setup_env.sh` — `cerberus` env, torch cu128.
   - ⚠️ 원본 `environment.yml`(torch 2.0.1+cu117)은 Blackwell sm_120 미지원 → 별도 env 필요.
2. **실행 스크립트:** `scripts/run_stage1.sh`, `run_stage2.sh`, `run_eval.sh`, `run_experiments.sh`, `extract_representations.py`.
3. **신규 실험 코드 스텁:** `experiments/` (E1·E3·효율성은 데이터 없이 합성 입력으로 즉시 검증 가능).
4. **데이터셋:** 마지막. WebVid(Stage1), HAWK 이상탐지 7종(Stage2), BG-critical 큐레이션.

---

## 5. 집필 매핑 (Paper file mapping)

| 파일 | 변경 |
|---|---|
| `paper_translation/improved/00_abstract.md` | 원리(CVD) 중심 재작성, HAWK는 한 줄 언급 |
| `01_introduction.md` | 한계→원리→기여(C1-C5). HAWK 의존 축소 |
| `02_related_work.md` | two-stream/분해/disentangled rep/scene context/video-LLM 대비. HAWK는 한 항목 |
| `03_methodology.md` | CVD 일반 정식화 → VAD 3-스트림 인스턴스화. 수식 일반 표기 |
| `04_experiments.md` | 실험 섹션: 프로토콜·E1-E4·표준·ablation·정성. 결과 placeholder |
| (신규) `05_experiment_appendix.md` | CDS/BSI 정식 정의, 진단 스위트 상세 |

**포맷 규칙:** `.claude/agent-memory/academic-paper-writer/feedback_paper_format.md` 준수
(이중언어 제목, 한글 학술체+영어 병기, blockquote 수식 `--- (N)`, Figure placeholder).
