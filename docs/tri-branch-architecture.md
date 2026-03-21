# Tri-Branch Architecture: Appearance + Motion + Background

## Overview

기존 Hawk 모델은 **Appearance(외형)**와 **Motion(움직임)** 2개 브랜치로 비디오를 분석했습니다.
여기에 **Background(배경)** 브랜치를 추가하여, 정적인 장면 맥락까지 학습하는 **Tri-Branch** 구조로 확장했습니다.

```
                        ┌─────────────────────┐
                        │    Input Video       │
                        └──────┬──────────────┘
               ┌───────────────┼───────────────┐
               ▼               ▼               ▼
      ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
      │  Original RGB  │ │ Motion Mask  │ │ Background Mask  │
      │   (전체 장면)   │ │ (움직이는 영역)│ │ (정적인 영역)     │
      └───────┬────────┘ └──────┬───────┘ └───────┬──────────┘
              ▼                 ▼                  ▼
      ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
      │   EVA-ViT      │ │ EVA-ViT      │ │ EVA-ViT          │
      │   (frozen)     │ │ (frozen)     │ │ (frozen)         │
      └───────┬────────┘ └──────┬───────┘ └───────┬──────────┘
              ▼                 ▼                  ▼
      ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
      │   Q-Former     │ │ Q-Former     │ │ Q-Former         │
      │   (frozen)     │ │ (frozen)     │ │ (frozen)         │
      └───────┬────────┘ └──────┬───────┘ └───────┬──────────┘
              ▼                 ▼                  ▼
      ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
      │ Video Q-Former │ │Video Q-Former│ │ Video Q-Former   │
      └───────┬────────┘ └──────┬───────┘ └───────┬──────────┘
              ▼                 ▼                  ▼
      ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐
      │  Projection    │ │ Projection   │ │ Projection       │
      │  (trainable)   │ │ (trainable)  │ │ (trainable)      │
      └───────┬────────┘ └──────┬───────┘ └───────┬──────────┘
              │                 │                  │
              └────────┬────────┴─────────┬────────┘
                       ▼                  ▼
               ┌──────────────┐   ┌──────────────┐
               │  Concat →    │   │ Individual   │
               │  LLaMA-2-7B  │   │ LLaMA loss   │
               └──────────────┘   └──────────────┘
```

---

## Motion vs Background: 상보적 관계

Background 브랜치는 Motion 브랜치의 **정확한 역(inverse)**입니다.

### 이미지(비전) 입력 비교

두 브랜치는 동일한 Optical Flow를 계산한 뒤, **마스크를 반대로** 적용합니다.

```python
# 공통: Farneback Optical Flow 계산
flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, ...)
mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

# Motion: 움직이는 영역만 보존 (mag > threshold)
mask_motion = (mag > mag_threshold).astype(np.uint8)
motion_frame = original_frame * mask_motion

# Background: 정적인 영역만 보존 (mag <= threshold)
mask_background = (mag <= mag_threshold).astype(np.uint8)
background_frame = original_frame * mask_background
```

시각적으로 표현하면:

| 원본 프레임 | Motion 프레임 | Background 프레임 |
|:---:|:---:|:---:|
| 전체 장면 | 움직이는 물체만 (나머지 검정) | 배경만 (움직이는 부분 검정) |

> `motion_frame + background_frame = original_frame` 관계가 성립합니다.

### 텍스트(캡션) 입력 비교

SpaCy NLP를 사용하여 텍스트도 상보적으로 분리합니다.

| | Motion 캡션 | Background 캡션 |
|---|---|---|
| **추출 대상** | 동사(VERB) + 주어 + 목적어 | 명사(NOUN/PROPN) + 형용사(ADJ) |
| **의미** | 행동/동작 서술 | 장면/환경 서술 |
| **예시** | "car crashes barrier" | "highway, dark, wet, road" |
| **함수** | `extract_actions_and_entities_sentence()` | `extract_background_entities_sentence()` |

예를 들어, 원본 캡션이 *"A car crashes into the barrier on a dark wet highway"* 인 경우:
- **Motion**: `"car crashes barrier"`
- **Background**: `"dark, wet, highway"`

---

## Loss 구성

총 5개의 loss를 조합하여 학습합니다.

```
Total Loss = L_appearance
           + 0.1 * L_motion
           + 0.1 * L_background
           + 0.1 * L_middle(appearance, motion)
           + 0.1 * L_middle(motion, background)
```

| Loss | 설명 |
|---|---|
| `L_appearance` | 원본 비디오 → LLaMA 생성 loss (메인) |
| `L_motion` | 움직임 프레임 → LLaMA 생성 loss |
| `L_background` | 배경 프레임 → LLaMA 생성 loss |
| `L_middle(app, motion)` | Appearance와 Motion의 중간 표현 cosine 유사도 loss |
| `L_middle(motion, bg)` | Motion과 Background의 중간 표현 cosine 비유사도 loss |

### Middle Loss의 의미

- **`L_middle(appearance, motion)`**: 외형과 움직임 표현이 **서로 보완적 정보**를 가지도록 유도
- **`L_middle(motion, background)`**: 움직임과 배경 표현이 **서로 다른 정보**를 인코딩하도록 강제 (상보적 관계 학습)

---

## 모델 컴포넌트 상세

### 각 브랜치의 구성 요소

| 컴포넌트 | Appearance | Motion | Background |
|---|---|---|---|
| Vision Encoder | `visual_encoder` | `visual_encoder_motion` | `visual_encoder_background` |
| LayerNorm | `ln_vision` | `ln_vision_motion` | `ln_vision_background` |
| Q-Former | `Qformer` | `Qformer_motion` | `Qformer_background` |
| Query Tokens | `query_tokens` | `query_tokens_motion` | `query_tokens_background` |
| Frame Pos Embed | `video_frame_position_embedding` | `video_frame_position_embedding_motion` | `video_frame_position_embedding_background` |
| Video Q-Former | `video_Qformer` | `video_Qformer_motion` | `video_Qformer_background` |
| Video Query Tokens | `video_query_tokens` | `video_query_tokens_motion` | `video_query_tokens_background` |
| Projection (Linear) | `llama_proj_0` | `llama_proj_motion_0` | `llama_proj_background_0` |
| Projection (Bottleneck) | `llama_proj_last` | `llama_proj_motion_last` | `llama_proj_background_last` |

### Freeze/Train 전략

- **Frozen** (학습하지 않음): EVA-ViT, Q-Former, LLaMA-2-7B
- **Trainable** (학습): Video Q-Former, Frame Position Embedding, Projection 레이어

---

## 데이터 파이프라인

### Stage 1: Pretrain (WebVid)

```
WebVid 비디오 → vis_processor()
   ├── load_video()           → clip (원본)
   ├── load_video_motion()    → clip_motion (움직임)
   └── load_video_background()→ clip_background (배경)

WebVid 캡션 → text_processor()
   ├── 원본 텍스트                    → text_input
   ├── extract_actions_and_entities() → text_input_motion
   └── extract_background_entities() → text_input_background
```

### Stage 2: Finetune (Hawk Dataset)

```
이상탐지 비디오 → load_video / load_video_motion / load_video_background
   ├── image          → 원본
   ├── image_motion   → 움직임
   └── image_background → 배경

QA 대화 데이터 → tokenizer
   └── image_token_len = num_video_query_token * 3  (기존 *2 → *3)
```

---

## 수정된 파일 목록

| 파일 | 변경 내용 |
|---|---|
| `hawk/processors/video_processor.py` | `compute_background()`, `load_video_background()` 추가, Processor에서 3개 반환 |
| `hawk/models/video_llama.py` | Background ViT, Q-Former, Video Q-Former, Projection 전체 추가 |
| `hawk/datasets/datasets/video_instruct_dataset.py` | Background 데이터 로딩, collater, token 수 변경 (x2→x3) |
| `hawk/datasets/datasets/webvid_datasets.py` | `extract_background_entities_sentence()`, background 데이터 반환 |
| `hawk/tasks/base_task.py` | `loss_background`, `mse_loss_bg` 추가, TensorBoard 로깅 |

---

## TensorBoard 모니터링

학습 중 다음 지표를 TensorBoard에서 확인할 수 있습니다:

| 지표 | 설명 |
|---|---|
| `Loss/total` | 전체 통합 loss |
| `Loss/ori` | Appearance 브랜치 loss |
| `Loss/motion` | Motion 브랜치 loss |
| `Loss/background` | Background 브랜치 loss |
| `Loss/middle` | Appearance-Motion 중간 표현 유사도 loss |
| `Loss/middle_bg` | Motion-Background 중간 표현 비유사도 loss |
| `Learning Rate` | 현재 학습률 |
