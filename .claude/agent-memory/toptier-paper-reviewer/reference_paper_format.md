---
name: reference-paper-format
description: CERBERUS 원고 포맷 규칙·파일 구성·심사 이력 문서 위치
metadata:
  type: reference
---

- **포맷 규칙 원본:** `.claude/agent-memory/academic-paper-writer/feedback_paper_format.md`
  (이중언어 제목 H1/H2/H3, 한글 학술체+영어 용어 병기, blockquote 수식 `> ... --- (N)`,
  Figure/Table placeholder).
- **원고 구성 (2026-08-11):** `paper_translation/improved/` 00_abstract, 01_introduction,
  02_related_work, 03_methodology, 04_experiments, **06_broader_impact**, 07_references,
  **08_appendix**(A~G). `05_*`(Results/Conclusion) 미작성 — **결과 부재는 지적 대상 아님**.
- **인용 번호 규칙:** HAWK 원본 [1]–[49] 승계(`origin/07_references.md`) + CERBERUS 추가분
  [50]–. 투고 시 재정렬 예정(작성자가 결과 확정 후로 보류). 2026-08-11 현재 [1]–[49] 중
  **22개가 미사용**이라 투고 목록에서 정리 필요.
- **심사 이력:** `docs/review-log.md`(라운드별 조치/미조치+이유+커밋), 전문은
  `docs/review-2026-08-10-methodology.md`.
- **실험 계획·결과별 대응:** `docs/experiment-roadmap.md`(S1/S2/S3 사다리, 부록 부활 조건).
- **주장↔근거 매핑:** `docs/evidence-index.md`(재현 명령 포함).
- **코드 정답지:** `hawk/processors/video_processor.py`(τ=0.2, `load_streams_aligned`),
  `hawk/tasks/base_task.py`(손실결합), `hawk/models/video_llama.py`(3브랜치),
  `hawk/models/projection.py`(병목 //16), `hawk/datasets/datasets/webvid_datasets.py`(언어추출),
  `experiments/`(CDS/BSI/loss_direction, bg_critical_benchmark).
