# 📚 Paper Translation — 논문 문서 색인

이 디렉터리에는 **두 개의 논문**이 들어 있습니다. 헷갈리지 않도록 구분하세요.

| 디렉터리 | 대상 | 설명 |
|---|---|---|
| [`origin/`](origin/) | **HAWK** (베이스, NeurIPS 2024) | 원본 HAWK 논문 **PDF 원문 + 한글 번역** (완성본) |
| [`improved/`](improved/) | **CERBERUS** (신규 연구) | 상보적 시각 분해(CVD) 원리를 제안하는 신규 논문 초안 (집필 중) |

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

**상보적 시각 분해(Complementary Visual Decomposition, CVD)** 원리를 제안하는 신규 논문. 원리 중심으로 서술하며, VAD는 검증 무대로 둔다. **현재 작업 대상**입니다. (설계·실험 마스터 플랜: [`../docs/cerberus-research-plan.md`](../docs/cerberus-research-plan.md))

| 파일 | 내용 | 상태 |
|---|---|---|
| `00_abstract.md` | 초록 — CVD 원리 + VAD 검증 | 초안 |
| `01_introduction.md` | 서론 — 패러다임 한계, CVD 원리, 기여 C1–C5 | 초안 |
| `02_related_work.md` | 관련 연구 — VAU, two-stream/decomposition, disentangled rep, scene context, video-LLM | 초안 |
| `03_methodology.md` | 방법론 — CVD 일반 정식화 → 3-스트림 인스턴스화, 방향성 분리 손실 | 초안 |
| `04_experiments.md` | 실험 — 프로토콜, 진단 스위트(CDS·BSI), E1–E4, 절제·분석·정성 (결과 placeholder) | 초안 |

### 집필 규칙
`improved/*.md`는 정해진 포맷을 따릅니다 (상세: [`../.claude/agent-memory/academic-paper-writer/feedback_paper_format.md`](../.claude/agent-memory/academic-paper-writer/feedback_paper_format.md)):
- 본문은 한글 학술체, 기술 용어는 첫 등장 시 영어 병기 — 예: 광학 흐름(optical flow)
- 제목 이중언어: `# N. Title (한글)`, `## N.N 한글 (English)`
- 수식은 blockquote(`>`) + 식 번호 `--- (N)`
- 그림 자리표시: `<!-- [Figure X 위치] -->`

> 번역은 `academic-paper-translator-ko`, 집필·구조화는 `academic-paper-writer` 에이전트를 사용하세요.
