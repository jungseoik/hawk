# vendor/ — third-party material bundled with this repo

Vendored (checked in) rather than installed, so a fresh `git clone` on any server has
the same tooling with no extra setup step. Nothing here is our work; see per-directory
attribution below and `../NOTICE.md` for the summary.

---

## academic-research-skills

| | |
|---|---|
| Upstream | https://github.com/Imbad0202/academic-research-skills |
| Author | Cheng-I Wu (© 2026) |
| License | **CC BY-NC 4.0** (Attribution–NonCommercial) — full text in `academic-research-skills/LICENSE` |
| Vendored at | 2026-08-09, shallow clone of `main` (`.git` removed) |
| Size | ~38 MB |

**License obligations we are meeting**: the upstream `LICENSE`, `NOTICE.md` and
`THIRD_PARTY.md` are kept verbatim inside the directory, the author is credited here and
in `../NOTICE.md`, and the material is unmodified. NonCommercial: this repo uses it for
writing an academic paper, which is not a commercial use. **If this repo is ever used to
build a commercial product, this directory must be removed first.**

### What it provides

Four skills (each `.claude/skills/<name>` is a relative symlink into this directory):

| Skill | Contents |
|---|---|
| `deep-research` | 13-agent research team, 8 modes (full research, quick brief, PRISMA systematic review, Socratic exploration, fact-check, literature review) |
| `academic-paper` | 12-agent writing pipeline, 11 modes (full paper, guided planning, outline, revision, abstract, format conversion, citation check, disclosure statements) |
| `academic-paper-reviewer` | 7-agent peer-review panel, 0–100 rubrics, journal-fit and devil's-advocate perspectives |
| `academic-pipeline` | orchestrator over all 10 stages with integrity-verification gates |

Plus three subagents symlinked into `.claude/agents/`: `research_architect_agent`,
`synthesis_agent`, `report_compiler_agent`.

### Why we brought it in

The gap it closes is **citation verification**, which none of our own agents do:
existence-checking against Semantic Scholar / OpenAlex / Crossref / arXiv, plus
claim-faithfulness audits (does the cited source actually support the sentence citing it?).
CERBERUS has no references section written yet, and fabricated citations are the fastest
way to lose a top-tier review.

### Relationship to our own agents (`.claude/agents/`)

They **overlap but do not replace each other**, so both are kept:

| Upstream | Ours | Why keep ours |
|---|---|---|
| `academic-paper` skill | `academic-paper-writer` | ours carries CERBERUS memory: paper format rules, C1–C5 framing, z_a/z_m/z_b notation |
| `academic-paper-reviewer` skill | `toptier-paper-reviewer` | ours carries the 2026-06-10 review: 6 recurring weaknesses + code-vs-paper ground truth |
| — | `experiment-design-strategist` | CERBERUS-specific ablation design vs the HAWK dual-branch baseline |

Rule of thumb: **use ours for anything CERBERUS-specific** (it already knows the
project's weak points), **use upstream for the generic machinery** — citation
verification, PRISMA literature sweeps, venue formatting, reviewer-panel rubrics.

### Updating

```bash
git clone --depth 1 https://github.com/Imbad0202/academic-research-skills.git /tmp/ars
rm -rf /tmp/ars/.git && rm -rf vendor/academic-research-skills
cp -a /tmp/ars vendor/academic-research-skills     # symlinks in .claude/ are relative → still valid
```

> Upstream also ships `hooks/` (`hooks.json`, `run_guard.sh`). Vendoring does **not**
> activate them — hooks only run when installed as a plugin. Leaving them inert is
> deliberate; enable them consciously if ever wanted.
