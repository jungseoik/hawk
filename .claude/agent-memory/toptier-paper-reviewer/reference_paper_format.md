---
name: reference-paper-format
description: CERBERUS 원고 포맷 규칙과 권위 있는 설계 출처 파일 위치
metadata:
  type: reference
---

- **포맷 규칙 원본:** `.claude/agent-memory/academic-paper-writer/feedback_paper_format.md` (이중언어 제목 H1/H2/H3, 한글 학술체+영어 용어 병기, blockquote 수식 `> ... --- (N)`, Figure/Table placeholder).
- **설계 정답지(논문 주장 기준):** `docs/cerberus-research-plan.md` — 원리 P1-P4, 기여 C1-C5, 실험 E1-E4 정의.
- **원고:** `paper_translation/improved/00_abstract.md ~ 04_experiments.md` (+ 예정 05_experiment_appendix.md).
- **코드 정답지:** `hawk/processors/video_processor.py`(τ=0.2, 통합플로우), `hawk/tasks/base_task.py`(손실결합), `hawk/models/video_llama.py`(3브랜치), `hawk/models/projection.py`(병목 //16), `hawk/datasets/datasets/webvid_datasets.py`(언어추출), `experiments/`(CDS/BSI/loss_direction 진단).
