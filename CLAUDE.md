# CLAUDE.md

이 파일은 이 저장소에서 작업하는 Claude(및 사람)를 위한 길잡이입니다. **작업 전에 반드시 먼저 읽으세요.**

---

## ⚠️ 가장 먼저 알아야 할 것: 이 저장소는 "두 개"로 구성됩니다

이 저장소에는 **베이스가 되는 기존 연구(HAWK)** 와 **그 위에 올라간 신규 연구(CERBERUS)** 가 함께 들어 있습니다. 코드와 문서가 한 곳에 섞여 있으므로, 어떤 파일이 어느 쪽에 속하는지 구분하는 것이 중요합니다.

| 구분 | **HAWK** (베이스) | **CERBERUS** (신규 연구 / 본 작업) |
|---|---|---|
| 정체 | NeurIPS 2024 공식 논문·코드 | HAWK를 확장한 **신규 논문** (개발명: HAWK++) |
| 아키텍처 | **Dual-Branch**: Appearance + Motion | **Tri-Branch**: Appearance + Motion + **Background** |
| 핵심 아이디어 | VLM에 모션 모달리티 명시적 통합 | Optical Flow 역마스크로 **정적 장면 맥락(Background)** 분리 학습 |
| 논문 원문 | `paper_translation/origin/hawk_neurips2024.pdf` (+ 한글 번역) | `paper_translation/improved/` (집필 중) |
| 상태 | 완성·공개됨 | **진행 중인 연구** |

> 한 줄 요약: **`Motion + Background = 원본 프레임`** 이라는 상보적 관계를 이용해, HAWK가 놓치던 정적 배경 맥락을 별도 브랜치로 학습하는 것이 CERBERUS의 핵심입니다.

---

## 📂 무엇이 어느 쪽인가 (파일 분류)

### HAWK (베이스 — 원본을 함부로 "원래 그랬다"고 단정하지 말 것)
- `hawk/` — 모델 코드. **원래는 Dual-Branch였고**, 아래 CERBERUS 작업으로 Background 브랜치가 추가됨
- `paper_translation/origin/hawk_neurips2024.pdf` — 원본 HAWK 논문 PDF (NeurIPS 2024)
- `paper_translation/origin/` — 원본 HAWK 논문 PDF + 한글 번역 (`00_abstract` ~ `08_appendix`)
- `app.py`, `train.py`, `configs/`, `environment.yml` — 원본 학습/데모 인프라

### CERBERUS (신규 연구 — 본 저장소에서 추가/수정된 부분)
- `paper_translation/improved/` — **신규 논문 CERBERUS 초안** (한글 학술체). 아키텍처의 권위 있는 출처
- `docs/tri-branch-architecture.md` — Tri-Branch(Background 브랜치) 아키텍처 상세 설계
- `improvements/optical_flow_and_loss_fix.md` — Optical Flow 중복 연산 제거 + Loss 방향 버그 수정 기록
- `hawk/` 내 다음 파일들이 CERBERUS를 위해 **수정/추가**됨:
  - `hawk/processors/video_processor.py` — `compute_motion_and_background()`, `load_video_motion_and_background()`
  - `hawk/models/video_llama.py` — Background ViT / Q-Former / Projection 추가
  - `hawk/datasets/datasets/video_instruct_dataset.py` — Background 로딩, token 수 ×2→×3
  - `hawk/datasets/datasets/webvid_datasets.py` — `extract_background_entities_sentence()`
  - `hawk/tasks/base_task.py` — `loss_background`, `mse_loss_bg`, TensorBoard 로깅

---

## 🧭 어디서부터 읽을까 (목적별 진입점)

| 하려는 일 | 먼저 볼 문서 |
|---|---|
| 저장소 전체 개요 / 설치 / 학습·데모 실행 | [`README.md`](README.md) |
| 신규 연구(CERBERUS)가 무엇인지 빠르게 파악 | [`paper_translation/improved/00_abstract.md`](paper_translation/improved/00_abstract.md), [`01_introduction.md`](paper_translation/improved/01_introduction.md) |
| Tri-Branch 구조·Loss·데이터 파이프라인 상세 | [`docs/tri-branch-architecture.md`](docs/tri-branch-architecture.md) |
| 최근 코드 개선 내역 (성능/버그 수정) | [`improvements/optical_flow_and_loss_fix.md`](improvements/optical_flow_and_loss_fix.md) |
| 원본 HAWK 이해 | [`paper_translation/origin/`](paper_translation/origin/) |
| 논문 문서 전체 색인 | [`paper_translation/README.md`](paper_translation/README.md) |

---

## ✍️ 작업 시 규칙

- **베이스 vs 신규 구분 유지**: 새 문서나 README를 쓸 때 HAWK(베이스)와 CERBERUS(신규)를 항상 명시적으로 구분하세요. 둘을 뭉뚱그리지 마세요.
- **논문 명칭**: 신규 논문의 공식 명칭은 **CERBERUS**입니다. 개발 과정/일부 메모에서는 **HAWK++**로도 불리므로 동일 대상임을 기억하세요.
- **논문 집필 포맷**: `paper_translation/improved/*.md`는 한글 학술체 + 영어 용어 병기, 이중언어 제목(H1/H2/H3), blockquote 수식 등 정해진 규칙을 따릅니다. 상세는 `.claude/agent-memory/academic-paper-writer/feedback_paper_format.md` 참고.
- **번역/집필 전용 에이전트**: 학술 번역은 `academic-paper-translator-ko`, 집필/구조화는 `academic-paper-writer` 에이전트를 사용하세요 (`.claude/agents/`).
- **코드 수정 시**: Background 관련 변경은 Motion 브랜치와의 **상보성(`Motion + Background = 원본`)** 을 깨지 않는지 확인하세요.

---

## 🛠️ 자주 쓰는 명령

```bash
# 환경 구성
apt install ffmpeg
conda env create -f environment.yml && conda activate hawk

# 학습 (stage1 pretrain / stage2 finetune)
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port='10000' \
  train.py --cfg-path ./configs/train_configs/stage1_pretrain.yaml
NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port='12001' \
  train.py --cfg-path ./configs/train_configs/stage2_finetune.yaml

# 데모
python app.py --cfg-path configs/eval_configs/eval.yaml --model_type llama_v2 --gpu-id 0
```
