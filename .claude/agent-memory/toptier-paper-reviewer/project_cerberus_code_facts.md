---
name: project-cerberus-code-facts
description: CERBERUS 코드/데이터에서 직접 검증한 손실·수식·분할·벤치마크 ground truth (논문 정합성 기준)
metadata:
  type: project
---

CERBERUS 논문 주장의 정답지(코드 실측). 검증일 2026-06-10 / 2026-08-11 / **2026-08-13(3차)**.

**손실 구조 (hawk/tasks/base_task.py L250-258):**
- 실제 결합: `loss = loss + 0.1*loss_motion + 0.1*loss_background + 0.1*mse_loss + 0.1*mse_loss_bg` — 항이 **5개가 아니라 사실상 4개 보조항**. 가중치 t_0=1, 나머지=0.1 일치.
- `mse_loss = 1 - cos(middle_result, middle_result_motion)` = 논문 L_sim (수식4) 일치.
- `mse_loss_bg = (1 + cos(middle_result_motion, middle_result_background))/2` = 논문 L_dis (수식5) 일치.
- **중대 불일치:** 손실 결합에 들어가는 것은 `loss_motion`, `loss_background`(=LLaMA CE loss)이지 논문이 말하는 L_ML/L_BL 언어손실 수식(8)과 표기/위치가 다름. 논문 수식10의 t_2·L_ML, t_3·L_BL ↔ 코드의 0.1*loss_motion, 0.1*loss_background 대응관계를 명시 안 하면 정합성 구멍. (3차 원고 §3.6 "구현 대응"으로 해소됨)
- **3차 신규 확인 — 코사인은 표본별이 아니다.** `base_task.py:260-262`가 세 `middle_result*`를
  `.view(1, -1)`로 **배치 전체를 하나의 벡터로 flatten**한 뒤 `F.cosine_similarity`를 부른다.
  즉 수식 (4)·(5)의 `cos(z_m, z_b)`는 배치·토큰·차원을 모두 편 **단일 스칼라**이며 표본별
  코사인이 아니다. 논문 §3.1.3/Appendix A는 표본별인 것처럼 서술 → 불일치. 이 사실은
  퇴화(모든 표본이 공유하는 상수 bias 회전으로 충족 가능)의 직접 원인이기도 하다.

**중대 불일치 2 — train_flag==0 경로 (video_llama.py L806):**
- 이 경로는 `loss_motion=loss_background=loss`(동일 메인손실). Stage-2 실효 목적은 `L_VL + 0.1 L_sim + 0.1 L_dis`. (3차 원고 §3.6·§4.1에 반영됨 — 해소)

**검증된 일치 항목:** τ=0.2, `bg_mask = 1 - motion_mask`, 통합 플로우 1회, 비디오 I/O 2회
(`load_streams_aligned`), `image_token_len = cur_token_len*3`, 병목 4096→256→4096,
장면언어 추출 규칙(수식7), 인코더 freeze 정책. 전부 코드와 일치.
- 언어 분할은 **서로소지만 의미적으로 깨끗하지 않다** — pobj/compound 명사가 장면으로 감
  (`barrier`, `highway`, `bat`). 3차 원고 §3.4가 이를 자인 → **해소**.

**CDS/BSI:** CDS 정의는 논문과 일치. BSI는 기본 구현이 **token-Jaccard**이고 문장 임베딩은
`embed_fn` 주입 시에만(`background_swap.py:43-56`). 원고 §4.2는 "본 실험은 문장 임베딩 코사인을
사용한다"고 단정 → 주입 코드가 확정되기 전까지 미검증 주장.

---

## 2차 추가 실측 (2026-08-11)

**HAWK(베이스)의 모션 스트림도 원본 픽셀 공간이다.** Farnebäck 플로우 크기를 [0,1] 정규화한
Mask × 원본 RGB. 따라서 CERBERUS의 실제 델타는 (a) soft→이진 마스크, (b) 보수 스트림
인스턴스화 두 가지뿐. → 3차 원고 §2.2에는 반영됐으나 **초록·§1.2·§3.1.1에는 미반영**.

**Background-critical 큐(600건)는 폐기됨.** 3차에서 heldout 471건으로 교체.

---

## 3차 추가 실측 (2026-08-13) — 결과 해석에 직결되는 것들

**heldout 471건은 학습 분할과 완전히 서로소다(오염 해소).**
`annotation_queue_heldout.json`(471) ∩ `all_videos_train.local.json`(7,066) = **0건**.
분할 구조: 전체 7,852 = train 7,066 + test 786, test 786 = heldout 471 + val 315.
test 도메인 분포: DoTA 494 / UCF_Crime 186 / UBnormal 54 / ShanghaiTech 40 / Ped1 7 / Ped2 3 / Avenue 2.
→ **UBnormal은 heldout 32 외에 val에 22건이 더 있다(합 54).** val은 체크포인트 선택에
쓰이지 않았으므로(고정 epoch 39 평가), UBnormal 기제 분석의 표본을 1.7배로 늘릴 수 있다.

**도메인별 scene-causal 라벨의 분모 문제(원고 표의 값이 바뀌는 사안).**
`labels_scene_causal_llm.json`(471건, LLM 라벨) 실측 분해:

| 도메인 | n | causal | incidental | no_scene | **normal** | causal/n | **causal/(이상 클립)** |
|---|---:|---:|---:|---:|---:|---:|---:|
| DoTA | 296 | 126 | 151 | 19 | 0 | 42.6% | **42.6%** |
| UCF_Crime | 112 | 10 | 66 | 22 | 14 | 8.9% | **10.2%** |
| UBnormal | 32 | 7 | 9 | 2 | 14 | 21.9% | **38.9%** |
| ShanghaiTech | 24 | 0 | 3 | 2 | 19 | 0% | **0/5** |
| Ped1/Ped2/Avenue | 7 | 0 | 1 | 1 | 5 | 0% | **0/2** |

→ 원고 §4.2·roadmap §2.5의 "UBnormal 22% / ShanghaiTech 0%(n=24)"는 **normal 클립을 분모에
포함한 값**이다. 이상 클립으로 정규화하면 UBnormal 38.9%로 DoTA 42.6%와 사실상 **매칭**되고,
ShanghaiTech의 "0%"는 n=5짜리 값이 된다. 이는 3-패턴 검정의 논증 구조를 바꾼다
(UBnormal↔DoTA가 causal 매칭·마스크 11.5배 차이의 자연 실험이 됨).
`no_scene` 46건은 검증 표본에서 6/6 사람이 뒤집었으므로 causal 약 8건이 더 있을 수 있음
(`agreement_scene_causal.json` notes).

**§4.8의 산술은 전부 맞다(재계산 확인).** 도메인별 (클립수 × 마스크비율) 합 = 144.0,
471로 나누면 0.306, DoTA 기여 132.9 → 0.282, 비중 **92.3%**. 평가분할 DoTA 62.8% / UCF 23.8%.

**주석 신뢰도(`agreement_scene_causal.json`):** n=80, **주석자 1인(`human_1`)**,
LLM=`gemini-3.1-pro-preview`. κ_4class=0.749, κ_causal-vs-rest=0.866, 관측일치 0.938.
파일에 이전 판의 `llm_causal_precision/recall` 철회 문구와 불일치 6건 감사 결과가 들어 있음.
**"사람 간 일치도"는 존재하지 않는다**(주석자 1인) — 원고 §4.2·Appendix H.4의 약속은 공허.

**구 스키마 주석의 출처(원고 §4.2 "무작위 추출" 근거).** `labels_human.json` 150건은
구 큐(`annotation_queue.json`, 600건, selection_group=candidate 107 / random_control 43 …)에
붙은 것이며, **그중 134건이 학습 분할 클립**이다. 설계 선택의 근거 통계이지 평가 수치는
아니지만, 같은 절이 "학습 분할은 어떤 이유로도 포함하지 않는다"고 선언하므로 출처 명시 필요.

**통계 스크립트 `scripts/compare_arms.py` 실측.**
쌍 부트스트랩(클립 단위) + Holm 구현은 존재하나, **Holm은 풀링된 3개 사전지정 대비에만**
적용된다(L124-140). 원고 §4.8이 **주 종점으로 사전 지정한 도메인별 값**(L142-154)에는
보정도 `verdict`도 없다. 즉 주 종점이 무보정, 보조 종점이 보정 — 방향이 뒤집혀 있다.

**Stage-2 정류점 이탈은 arm마다 다를 수 있다(통제 위협).** 2026-08-13 실측:
- `runs/abl_flow/train.log`: 40 epoch 완주, 최종 `middleloss ≈ 0.039–0.050`,
  `middleloss_bg ≈ 0.044–0.058` → **이탈 상태로 종료**.
- `runs/abl_duplicate/train.log`: 15 epoch 시점 `middleloss = middleloss_bg = 0.0000000`
  (아직 정류점).  `runs/abl_random_mask`: 5 epoch.
→ 이탈이 arm 특이적이면 "정적 입력만 다르다"는 4-arm 통제의 전제가 실현 런에서 깨진다.
**arm별 이탈 epoch를 반드시 기록·보고할 것.**

**`configs/train_configs/ablation/stage2_flow_reinit.yaml` 존재 확인.** `flow`와의 차이는
`model.static_reinit: True` 한 줄뿐(+주석). 즉 입력은 flow, 정적 브랜치 학습가능부만 초기화.
