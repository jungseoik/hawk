---
name: Paper Format Conventions
description: Specific formatting rules for the HAWK++ improved paper markdown files
type: feedback
---

The user requires a specific markdown format for the improved paper sections:
- H1: `# N. Title (한글제목)` with both English and Korean
- H2: `## N.N 한글제목 (English Title)` Korean-first with English in parentheses
- H3: `### 한글소제목 (English)` same pattern
- Body text entirely in Korean, technical terms with English in parentheses on first use
- Math formulas in blockquote (`>`) format with equation numbers `--- (N)`
- Figure placeholders: `<!-- [Figure X 위치] -->`
- Scholarly, formal tone (학술적 격식체)

**Why:** Consistent formatting across all paper sections for eventual compilation.
**How to apply:** Follow these conventions for every `paper_translation/improved/*.md` file.
