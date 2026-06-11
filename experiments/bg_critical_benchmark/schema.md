# Background-Critical Diagnostic Benchmark — Annotation & Manifest Schema

> Paper experiment **E2** (see `../../docs/cerberus-research-plan.md` §3 and the
> Experiments section of the paper). Data is collected **last**; this file fixes
> the protocol and data contract so curation/annotation can start immediately.

## 1. Purpose

Isolate anomalies whose **judgment is determined by the static scene context**,
not by motion. A model with a strong static stream should win *here* even if it
ties a motion-only model elsewhere — making the benchmark a falsifiable test of
the Complementary Visual Decomposition (CVD) claim.

## 2. Two construction routes

### Route A — Curated subset (real videos)
Filter existing VAD datasets (DoTA, UCF-Crime, ShanghaiTech, UCSD) and tag each
clip with `motion_sufficiency`:

- `motion_sufficient`   : anomaly identifiable from motion alone.
- `context_dependent`   : anomaly requires static scene context (road condition,
  location, weather, signage, stationary-object-in-dangerous-place).
- `context_critical`    : motion is normal/absent; ONLY background reveals the
  anomaly (e.g. person standing on a highway).

The benchmark = `context_dependent` ∪ `context_critical`.

### Route B — Counterfactual background swap (controlled, causal)
Composite the **same moving foreground** onto **different backgrounds** to hold
motion fixed while flipping the scene. Measures whether the model *uses* the
background causally (see `background_swap.py`).

- pair = (foreground clip, {safe_bg, dangerous_bg})
- expected: anomaly judgment / description should change with the background.

## 3. Manifest JSON (one record per clip)

See `manifest_template.json`. Fields:

| field | type | meaning |
|---|---|---|
| `clip_id` | str | unique id |
| `source_dataset` | str | DoTA / UCF-Crime / ... / synthetic-swap |
| `video_path` | str | path to clip |
| `motion_sufficiency` | enum | motion_sufficient / context_dependent / context_critical |
| `anomaly` | bool | is this clip anomalous |
| `gt_description` | str | reference description |
| `scene_tokens` | list[str] | background/scene words (the static-language GT) |
| `motion_tokens` | list[str] | action words (the dynamic-language GT) |
| `swap_group` | str/null | route-B pairing key (null for route A) |
| `swap_role` | enum/null | foreground / safe_bg / dangerous_bg |

## 4. Metric — Background Sensitivity Index (BSI)

For a route-B swap group with the same foreground on `safe_bg` vs `dangerous_bg`:

```
BSI = 1 - sim( response(fg ⊕ safe_bg), response(fg ⊕ dangerous_bg) )
```

where `sim` is sentence-embedding cosine (or BLEU). **High BSI** = the model's
output is sensitive to the background, i.e. it *uses* the static stream.
A motion-only model is expected to have BSI ≈ 0 (background-blind).

Report: mean BSI, and accuracy on `context_critical` subset vs the full set.
