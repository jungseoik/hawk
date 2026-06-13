# 📐 docs/ — CERBERUS 기술 문서

이 디렉터리의 문서는 모두 **신규 연구 CERBERUS** (HAWK의 Tri-Branch 확장)에 관한 것입니다.
베이스 HAWK와의 구분은 [`../CLAUDE.md`](../CLAUDE.md)를 참고하세요.

| 문서 | 내용 |
|---|---|
| [`reproduce.md`](reproduce.md) | **⭐ git clone → 학습까지 재현 가이드** — 환경(Blackwell)·가중치·데이터(다운로드·추출)·스모크·학습·평가 단계별 + 트러블슈팅 |
| [`tri-branch-architecture.md`](tri-branch-architecture.md) | Tri-Branch(Appearance + Motion + **Background**) 아키텍처 전체 설계 — 브랜치 구조, 상보적 Loss, 데이터 파이프라인, 수정 파일 목록, TensorBoard 지표 |
| [`cerberus-research-plan.md`](cerberus-research-plan.md) | 논문 reframing(CVD 원리)·기여(C1–C5)·실험(E1–E4) 마스터 플랜 |
| [`data_webvid_setup.md`](data_webvid_setup.md) | **WebVid 데이터 준비 재현 절차** — 10M 소스(jxie)에서 다운로드·추출·분산·학습 연결, 함정/용량/검증 기록 |
| [`../improvements/optical_flow_and_loss_fix.md`](../improvements/optical_flow_and_loss_fix.md) | 코드 개선 기록 — Optical Flow 중복 연산 제거(~50% 단축), Motion-Background 비유사성 Loss 방향 버그 수정 |

> 논문 초안은 [`../paper_translation/improved/`](../paper_translation/improved/)에 있습니다.
