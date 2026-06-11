---
name: project-cerberus-code-facts
description: CERBERUS 코드에서 직접 검증한 손실/수식/하이퍼파라미터 ground truth (논문 정합성 기준)
metadata:
  type: project
---

CERBERUS 논문 주장의 정답지(코드 실측). 검증일 2026-06-10.

**손실 구조 (hawk/tasks/base_task.py L250-258):**
- 실제 결합: `loss = loss + 0.1*loss_motion + 0.1*loss_background + 0.1*mse_loss + 0.1*mse_loss_bg` — 항이 **5개가 아니라 사실상 4개 보조항**. 가중치 t_0=1, 나머지=0.1 일치.
- `mse_loss = 1 - cos(middle_result, middle_result_motion)` = 논문 L_sim (수식4) 일치.
- `mse_loss_bg = (1 + cos(middle_result_motion, middle_result_background))/2` = 논문 L_dis (수식5) 일치.
- **중대 불일치:** 손실 결합에 들어가는 것은 `loss_motion`, `loss_background`(=LLaMA CE loss)이지 논문이 말하는 L_ML/L_BL 언어손실 수식(8)과 표기/위치가 다름. 논문 수식10의 t_2·L_ML, t_3·L_BL ↔ 코드의 0.1*loss_motion, 0.1*loss_background 대응관계를 명시 안 하면 정합성 구멍.

**중대 불일치 2 — train_flag==0 경로 (video_llama.py L806):**
- 이 경로는 `return {"loss": loss, "loss_motion": loss, "loss_background": loss, ...}` — **loss_motion=loss_background=loss(동일 메인손실)**. 즉 이 경로(Stage2 추정)에서는 L_ML/L_BL이 독립 언어손실이 아니라 메인손실 복제. 논문은 두 단계 모두 독립 언어감독처럼 서술 → 불일치.
- 별도 경로(L950)에서만 loss_motion/loss_background가 text_input_motion/background로 독립 CE 계산.

**검증된 일치 항목:**
- τ(mag_threshold)=0.2 (video_processor.py L25) — 논문 수식3 일치.
- bg_mask = 1 - motion_mask (video_processor.py L47) — 논문 수식9 일치.
- 통합 플로우: flow 1회→두 마스크 동시 (compute_motion_and_background) — C4 일치. 단 비디오 I/O는 load_video + load_video_motion_and_background로 **2회**(원본1+통합1), 논문 표 "3회→2회"와 일치.
- image_token_len = cur_token_len*3 (video_instruct_dataset.py L284) — ×3 일치.
- 병목 4096→256→4096: Projection(projection.py): encoder_0(h→h), encoder_1(h→h//16), decoder_2(h→h). h=4096이면 256. 단 **encoder_1 입력은 h(4096)이고 출력 h//16(256)**; decoder_2는 x_full(h)을 받음. 논문 "encoder_1: hidden→hidden/16" 일치, 압축표현 z=x_compress=encoder_1 출력 일치.
- 장면언어 추출 (webvid_datasets.py L53-55): NOUN/PROPN & dep∉{nsubj,nsubjpass,dobj} + ADJ — 논문 수식7 일치. 단 **pobj/obj가 동적쪽 objects엔 포함(L38)되나 정적 제외조건엔 dobj만** → 미묘한 비대칭(pobj 명사가 정적으로 샐 수 있음). 검증 필요 지점.
- 인코더 freeze: ViT/ln_vision/Qformer/query_tokens frozen, video_Qformer 학습 (video_llama.py L92-157) — 논문 파이프라인 서술 일치.

**CDS/BSI 정의 (experiments/):**
- CDS (disentanglement.py L138-153): Coverage=CKA([z_m;z_b],z_a), Redundancy=CKA(z_m,z_b), CDS=Coverage*(1-Redundancy) — 논문 수식11-13 일치. 합성 3체제(complementary/partial/collapsed) 검증 코드 존재.
- BSI (background_swap.py L43): 1 - sim(R_safe, R_danger), 기본 sim=token-Jaccard(embed_fn 없으면), 논문은 "문장임베딩 코사인"이라 단정 → **기본구현은 Jaccard**라 불일치. composite = m*fg + (1-m)*bg 일치.
- loss_direction.py: buggy(1-cos)→cos→+1 붕괴, fixed((1+cos)/2)→cos→-1 분리. 단 코드 주석이 "cosine gradient vanishes near ±1"이라 정확수렴 아닌 부호(sign)만 주장.
