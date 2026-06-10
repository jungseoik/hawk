# 📚 Paper Translation — 논문 문서 색인

이 디렉터리에는 **두 개의 논문**이 들어 있습니다. 헷갈리지 않도록 구분하세요.

| 디렉터리 | 대상 | 설명 |
|---|---|---|
| [`origin/`](origin/) | **HAWK** (베이스, NeurIPS 2024) | 원본 HAWK 논문 **PDF 원문 + 한글 번역** (완성본) |
| [`improved/`](improved/) | **CERBERUS** (신규 연구) | HAWK를 Tri-Branch로 확장한 신규 논문 초안 (집필 중) |

> 저장소 전체 구조와 베이스 vs 신규 구분은 [`../CLAUDE.md`](../CLAUDE.md) 참고.

---

## 🟦 origin/ — 원본 HAWK 논문 (PDF + 한글 번역)

원본 HAWK 논문의 PDF 원문과 이를 한글 학술체로 번역한 문서. **참조용**이며 수정 대상이 아닙니다.

| 파일 | 내용 |
|---|---|
| 📄 `hawk_neurips2024.pdf` | **원본 논문 PDF 원문** (NeurIPS 2024) |
| `00_abstract.md` | 초록 |
| `01_introduction.md` | 서론 |
| `02_related_work.md` | 관련 연구 |
| `03_data_engineering.md` | 데이터 엔지니어링 |
| `04_methodology.md` | 방법론 (Dual-Branch: Appearance + Motion) |
| `05_experiments.md` | 실험 |
| `06_conclusion.md` | 결론 |
| `07_references.md` | 참고문헌 |
| `08_appendix.md` | 부록 |

## 🟩 improved/ — CERBERUS 신규 논문 (집필 중)

HAWK에 **Background 브랜치**를 추가한 Tri-Branch 신규 논문. **현재 작업 대상**입니다.

| 파일 | 내용 | 상태 |
|---|---|---|
| `00_abstract.md` | 초록 — CERBERUS 제안 | 초안 |
| `01_introduction.md` | 서론 — Background 브랜치 동기, 기여 3가지 | 초안 |
| `02_related_work.md` | 관련 연구 — VAD, 비디오 대규모 모델, 장면 맥락 | 초안 |
| `03_methodology.md` | 방법론 — Tri-Branch 구조, 상보적 Loss | 초안 |
| `04_experiment_plan.md` | 실험 계획 | 초안 |

### 집필 규칙
`improved/*.md`는 정해진 포맷을 따릅니다 (상세: [`../.claude/agent-memory/academic-paper-writer/feedback_paper_format.md`](../.claude/agent-memory/academic-paper-writer/feedback_paper_format.md)):
- 본문은 한글 학술체, 기술 용어는 첫 등장 시 영어 병기 — 예: 광학 흐름(optical flow)
- 제목 이중언어: `# N. Title (한글)`, `## N.N 한글 (English)`
- 수식은 blockquote(`>`) + 식 번호 `--- (N)`
- 그림 자리표시: `<!-- [Figure X 위치] -->`

> 번역은 `academic-paper-translator-ko`, 집필·구조화는 `academic-paper-writer` 에이전트를 사용하세요.
