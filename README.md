<div align="center">

# CERBERUS — Three-Headed Video Understanding for Open-World Anomaly Detection

<img src="figs/icon.png" alt="Have eyes like a HAWK!" width="80">

**HAWK(NeurIPS 2024)를 확장한 신규 연구 저장소**

</div>

---

## 📌 이 저장소는 무엇인가 (먼저 읽어주세요)

이 저장소는 **두 가지 연구**가 함께 들어 있습니다. 코드와 문서가 한곳에 섞여 있으므로, 무엇이 어느 쪽인지 먼저 구분하세요.

| | **HAWK** (베이스) | **CERBERUS** (이 저장소의 신규 연구) |
|:---|:---|:---|
| **정체** | NeurIPS 2024 공식 논문·코드 | HAWK를 확장한 **신규 논문** (개발명 HAWK++) |
| **아키텍처** | **Dual-Branch** — Appearance + Motion | **Tri-Branch** — Appearance + Motion + **Background** |
| **새 아이디어** | VLM에 모션 모달리티 통합 | Optical Flow 역마스크로 **정적 장면 맥락(Background)** 분리 학습 |
| **논문** | 📄 [원문 PDF](paper_translation/origin/hawk_neurips2024.pdf) · [한글 번역](paper_translation/origin/) | [`paper_translation/improved/`](paper_translation/improved/) *(집필 중)* |
| **상태** | 완성·공개 | **진행 중** |

> **핵심 한 줄**: `Motion + Background = 원본 프레임`이라는 상보적 관계를 이용해, HAWK가 놓치던 **정적 배경 맥락**(도로 상태, 주변 환경)을 별도 브랜치로 학습합니다.

> 코드·문서 작업 시 길잡이는 [`CLAUDE.md`](CLAUDE.md)를 참고하세요.

---

## 🗺️ 저장소 지도

```
hawk/
├── CLAUDE.md                  ← 작업 길잡이 (베이스 vs 신규 구분, 진입점)
├── README.md                  ← (이 파일) 전체 개요
│
├── 🟦 HAWK (베이스)
│   ├── paper_translation/origin/
│   │   ├── hawk_neurips2024.pdf   원본 논문 PDF (NeurIPS 2024)
│   │   └── 00~08_*.md             원본 논문 한글 번역
│   ├── app.py / train.py      데모·학습 진입점
│   └── configs/               학습·평가 설정
│
├── 🟩 CERBERUS (신규 연구)
│   ├── paper_translation/improved/      신규 논문 초안 (한글)
│   ├── docs/tri-branch-architecture.md  Tri-Branch 아키텍처 상세
│   └── improvements/                    코드 개선 기록 (성능·버그 수정)
│
└── hawk/                      모델 코드 (★원래 Dual-Branch → 현재 Background 브랜치 추가됨)
```

> 📚 **문서 색인**: [`paper_translation/README.md`](paper_translation/README.md) · **신규 아키텍처**: [`docs/tri-branch-architecture.md`](docs/tri-branch-architecture.md) · **개선 내역**: [`improvements/`](improvements/)

---

## 🟩 CERBERUS: Tri-Branch Architecture (신규 연구)

기존 HAWK의 **Appearance + Motion** 2-브랜치 구조에 **Background 브랜치**를 추가하여, 비디오를 3가지 관점에서 분석합니다.

| Branch | 이미지 입력 | 텍스트 입력 | 역할 |
|:---:|:---:|:---:|:---:|
| **Appearance** | 원본 RGB 프레임 | 전체 캡션 | 전체 장면 이해 |
| **Motion** | 움직이는 영역만 (Optical Flow `mag > threshold`) | 동사 + 주어 + 목적어 | 이상 행동 포착 |
| **Background** | 정적인 영역만 (Optical Flow `mag <= threshold`) | 명사 + 형용사 | 장면 맥락 이해 |

> Motion과 Background는 Optical Flow 마스크의 **정확한 역(inverse)** 관계입니다. → `motion_frame + background_frame = original_frame`

```
Input Video ──┬── Original RGB ──→ Appearance Branch ──┐
              ├── Motion Mask ───→ Motion Branch ──────┤──→ Concat → LLaMA-2-7B
              └── Background Mask → Background Branch ─┘
```

**Loss 구성:**
```
Total Loss = L_appearance + 0.1·L_motion + 0.1·L_background
           + 0.1·CosineSim(appearance, motion)      # 유사성 강제
           + 0.1·CosineSim(motion, background)       # 비유사성 강제
```

> 상세 아키텍처·Loss·데이터 파이프라인: [`docs/tri-branch-architecture.md`](docs/tri-branch-architecture.md)
> 최근 개선(Optical Flow 중복 연산 제거 ~50% 단축, Loss 방향 버그 수정): [`improvements/optical_flow_and_loss_fix.md`](improvements/optical_flow_and_loss_fix.md)

---

## 🟦 HAWK: 베이스 (NeurIPS 2024)

> 원본 저장소: [Hawk](https://openreview.net/pdf?id=vBKoEZ1PG3) · [HuggingFace Demo](https://huggingface.co/spaces/Jiaqi-hkust/hawk) · [Dataset](https://huggingface.co/datasets/Jiaqi-hkust/hawk)

[Jiaqi Tang^](https://jqt.me/), [Hao Lu^](https://scholar.google.com/citations?user=OrbGCGkAAAAJ&hl=zh-TW), [Ruizheng Wu](https://scholar.google.com/citations?user=OOagpAcAAAAJ&hl=en), [Xiaogang Xu](https://xuxiaogang.com/), [Ke Ma](https://scholar.google.com.hk/citations?user=yXGNGS8AAAAJ&hl=en), Cheng Fang, [Bin Guo](http://www.guob.org/), [Jiangbo Lu](https://sites.google.com/site/jiangbolu), [Qifeng Chen](https://cqf.io/), [Ying-Cong Chen*](https://www.yingcong.me/)

### 🔍 Motivation — Have eyes like a Hawk!
- 🚩 기존 VAD 시스템은 장면에 대한 피상적 의미 이해와 최소한의 사용자 상호작용에 머무는 경우가 많음
- 🚩 데이터 부족으로 인해 open-world 시나리오 적용이 제한됨

<div align="center"><img src="figs/motivation1.png" alt="Hawk"></div>

---

## ▶️ Getting Started

> 🚀 **재현(Reproduce) 전체 경로**: git clone → 환경 → 가중치 → 데이터(다운로드·추출) → 스모크 → 학습 → 평가까지 한 번에 따라할 수 있는 단일 가이드는 **[`docs/reproduce.md`](docs/reproduce.md)** 를 보세요. 아래는 요약입니다.

### 🪒 Installation

> ⚠️ 업스트림 `environment.yml`은 `torch 2.0.1+cu117`로 고정되어 **NVIDIA Blackwell(sm_120)에서 학습이 실행되지 않습니다.** 최신/Blackwell GPU에서는 아래 Blackwell 호환 스크립트를 사용하세요.

```bash
# (권장) Blackwell 호환 — cerberus env (torch cu128) + 전체 학습 스택
bash scripts/setup_env.sh --full
conda activate cerberus
pip install -e .

# (구형 GPU·원본 재현용) HAWK 원본 환경
# apt install ffmpeg && conda env create -f environment.yml && conda activate hawk
```

### 🏰 Pretrained / Fine-tuned Model

| Checkpoint | Link | Note |
|:---|:---|:---|
| Video-LLaMA-2-7B-Finetuned | [link](https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned/tree/main) | 학습 초기 가중치 |
| **Hawk_Pretrained** | [link](https://huggingface.co/Jiaqi-hkust/hawk) | [WebVid](https://github.com/m-bain/webvid)로 사전학습 |
| **Hawk_Finetuned** | [link](https://huggingface.co/Jiaqi-hkust/hawk) | [Hawk dataset](https://huggingface.co/datasets/Jiaqi-hkust/hawk)로 파인튜닝 |

- 사전학습 모델만 쓰려면 **Hawk_Pretrained**, 이상 이해까지 활용하려면 **Hawk_Finetuned**를 사용하세요.

---

## ⏳ Demo

[`configs/eval_configs/eval.yaml`](configs/eval_configs/eval.yaml)에서 경로를 본인 환경에 맞게 교체하세요.
```yaml
# LLaMA-2-chat 베이스 (Video-LLaMA-2-7B-Finetuned에서 다운로드 가능)
llama_model: ".../Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf"
# Hawk 가중치 (Pretrained 또는 Finetuned)
ckpt: '.../checkpoint.pth'
```
```bash
python app.py --cfg-path configs/eval_configs/eval.yaml --model_type llama_v2 --gpu-id 0
```
- GUI [Online Demo](https://huggingface.co/spaces/Jiaqi-hkust/hawk)

<div align="center"><img src="figs/demo.png" alt="Hawk"></div>

---

## 🖥️ Training

> 전체 재현 절차(환경·가중치·데이터·스모크·학습)는 **[`docs/reproduce.md`](docs/reproduce.md)** 에 단계별로 정리되어 있습니다. 아래는 데이터/명령 요약입니다.

### 💾 Stage 1 데이터: WebVid (사전학습)
원본 WebVid는 배포 중단되어, 영상 바이트가 내장된 미러 `jxie/webvid_10m`(WebVid-10M)에서 받아 개별 mp4로 추출합니다. 다운로드(stall 자동복구)·추출·용량·함정까지 재현 절차는 **[`docs/data_webvid_setup.md`](docs/data_webvid_setup.md)** 참고.
```bash
# 다운로드(~2M) + 추출(개별 mp4) — 상세/규모선택은 위 문서
python scripts/resilient_hf_download.py --repos jxie/webvid_10m jxie/webvid_10m_part_0 jxie/webvid_10m_part_1 \
  --base /data/pia --include "data/*.parquet" --stall 300 --min-mbps 3
python scripts/build_webvid_split.py --single /data/pia/webvid_extracted
```

### 💾 Stage 2 데이터: 이상 탐지 (HAWK 7-dataset)
- Hawk 데이터셋(비디오 + 어노테이션): [HuggingFace DOWNLOAD](https://huggingface.co/datasets/Jiaqi-hkust/hawk)
- 어노테이션만: [Google Drive DOWNLOAD](https://drive.google.com/file/d/1WCnizldWZvtS4Yg5SX7ay5C3kUQfz-Eg/view?usp=sharing)
- 원 출처: [CUHK_Avenue](https://www.cse.cuhk.edu.hk/leojia/projects/detectabnormal/dataset.html), [DoTA](https://github.com/MoonBlvd/Detection-of-Traffic-Anomaly), [Ped1/Ped2](http://www.svcl.ucsd.edu/projects/anomaly/dataset.htm), [ShanghaiTech](https://svip-lab.github.io/dataset/campus_dataset.html), [UBNormal](https://github.com/lilygeorgescu/UBnormal/), [UCF_Crime](https://www.crcv.ucf.edu/projects/real-world/)

데이터 구조:
```
(Hawk_data)
Annotation
├── All_Mix
│   ├── all_videos_all.json / all_videos_test.json / all_videos_train.json
├── CUHK_Avenue/Avenue.json
├── DoTA/DoTA.json
├── ... └── UCF_Crime/...
Videos
├── CUHK_Avenue/ ├── DoTA/ ├── Ped1/ ...
```
> `All_Mix`는 학습·테스트 전체 데이터셋을 합친 디렉터리입니다. 데이터 경로는 재정의가 필요합니다.

### 🔨 Configuration
[`configs/train_configs`](configs/train_configs)의 2-stage 설정에서 경로를 교체하세요.
```yaml
llama_model: ".../Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf"
# stage1 사전학습 후 vision branch 체크포인트 (stage2 전용)
ckpt: ".../checkpoint.pth"
```

### 🧪 스모크 테스트 (본 학습 전 점검)
```bash
conda run -n cerberus python scripts/smoke_test.py --frames 8                 # 모델 빌드 + 더미 forward
torchrun --nproc_per_node=1 train.py --cfg-path configs/train_configs/stage1_smoke.yaml  # 5 iters 실학습
```

### 🖥️ To Train
```bash
# 사전학습 (Stage 1)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port='10000' \
  train.py --cfg-path ./configs/train_configs/stage1_pretrain.yaml

# 파인튜닝 (Stage 2) — stage2 config의 ckpt:를 Stage1 출력으로 지정
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port='12001' \
  train.py --cfg-path ./configs/train_configs/stage2_finetune.yaml
```
*검증 환경: 2 × RTX PRO 6000 (Blackwell, 96G), batch 1·32프레임·3스트림 ≈ 62GB VRAM. 원본 HAWK는 4 × A6000 48G.*

### 📊 Tri-Branch 학습 (CERBERUS)
각 비디오에서 3종류 입력이 자동 생성됩니다:
```
1. Original RGB                       → Appearance branch
2. Optical Flow (mag > 0.2) masking   → Motion branch (움직이는 영역만)
3. Optical Flow (mag <= 0.2) masking  → Background branch (정적 영역만)
```
TensorBoard 모니터링 지표: `Loss/total`, `Loss/ori`, `Loss/motion`, `Loss/background`, `Loss/middle`(app-motion), `Loss/middle_bg`(motion-background)

> 상세: [`docs/tri-branch-architecture.md`](docs/tri-branch-architecture.md)

---

## 🌐 Citation

**HAWK (베이스 논문):**
```latex
@inproceedings{atang2024hawk,
  title = {Hawk: Learning to Understand Open-World Video Anomalies},
  author = {Tang, Jiaqi and Lu, Hao and Wu, Ruizheng and Xu, Xiaogang and Ma, Ke and Fang, Cheng and Guo, Bin and Lu, Jiangbo and Chen, Qifeng and Chen, Ying-Cong},
  year = {2024},
  booktitle = {Neural Information Processing Systems (NeurIPS)}
}
```
> CERBERUS는 집필 중인 신규 연구로, 인용 정보는 추후 추가됩니다.

## 📜 Acknowledgment
이 프로젝트는 [Video-LLaMA](https://github.com/DAMO-NLP-SG/Video-LLaMA)와 [HAWK](https://github.com/jqtangust/hawk)에서 영감을 받았습니다.
HAWK 원본 논문 관련 문의: `jtang092@connect.hkust-gz.edu.cn`
