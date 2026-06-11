---
name: User Research Context
description: User writes the CERBERUS paper (principle-first CVD for video-language understanding), Korean academic style, top-tier target
type: user
---

User is writing **CERBERUS**, a top-tier-targeted paper (CVPR/NeurIPS/ICCV) in Korean academic style (한글 학술체) with English terms in parentheses. Files live in `paper_translation/improved/` as numbered markdown (00_abstract.md, 01_introduction.md, ...).

**Framing (authoritative — set by `docs/cerberus-research-plan.md`):** As of 2026-06-10 the paper was reframed from a "HAWK Dual-Branch + Background = Tri-Branch" derivative narrative to a **principle-first** narrative centered on **Complementary Visual Decomposition (CVD)** — a general principle that losslessly partitions a frame into complementary dynamic/static streams in original pixel space (vs. lossy two-stream / optical-flow-image / RGB-diff). HAWK is now positioned only as a special case / one baseline (≤1 sentence in abstract, ≤1 paragraph elsewhere). VAD is framed as a *testbed*, not the scope. Do NOT use "extends HAWK / adds a branch to HAWK" phrasing.

**Contributions are C1–C5 (principle-first):** C1 lossless complementary pixel-space decomposition; C2 directional asymmetric disentanglement objective + information-theoretic justification; C3 modality-matched linguistic supervision; C4 unified flow computation; C5 diagnostic suite (CDS metric + Background-critical benchmark).

**Notation convention:** appearance `z_a`, dynamic/motion `z_m`, static/background `z_b`. Similarity loss `L_sim = 1 − cos(z_a,z_m)`; dissimilarity `L_dis = (1+cos(z_m,z_b))/2`. Total loss weights `t_0=1, t_1..t_4=0.1`. See [[Paper Format Conventions]] for markdown rules.

**Hard rule:** never fabricate experimental numbers — methodology only; performance claims go in `<!-- 결과 추가 예정 -->` placeholders.
