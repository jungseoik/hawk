#!/usr/bin/env python
"""이상 이해 성능 배치 평가 — 설명 생성(A)과 질의 응답(B).

레포에 BLEU/GPT-guided 점수를 계산하는 코드가 없어서, 학습이 끝나도 논문 Table 1 을
채울 수단이 없었다. 이 스크립트가 그 구멍을 메운다.

설계 원칙 세 가지:

1. **결정론.** 생성은 greedy(`do_sample=False`)로 고정한다. 표본 추출을 켜면 같은
   입력에도 응답이 달라지고, 그 변동이 BSI(배경 교체 전후 응답 차이)에 양의 편향으로
   들어가 "배경을 인과적으로 활용한다"는 결론을 인공적으로 만들 수 있다.
2. **arm 간 비교 가능성.** 프롬프트·프레임 수·최대 토큰·시드가 전부 고정되며 결과
   파일에 함께 기록된다. 두 arm 의 수치를 비교하려면 이 값들이 같아야 한다.
3. **판정자 분리.** BLEU 는 의존성 없이 항상 계산한다. GPT-guided 3종
   (Reasonability / Detail / Consistency)은 외부 LLM 판정자가 필요하므로 기본은
   꺼져 있고, `--judge` 로 명시적으로 켠다. 판정자 없이 나온 결과를 GPT-guided 로
   보고해서는 안 된다.

사용법:
    $CERBERUS_PY scripts/evaluate.py \
        --cfg configs/eval_configs/eval.yaml \
        --ckpt /home/work/seoik/runs/abl_flow/main/checkpoint_39.pth \
        --out experiments/out/eval_abl_flow.json --limit 0

    # 여러 arm 을 한 번에 비교
    $CERBERUS_PY scripts/evaluate.py --compare experiments/out/eval_*.json
"""
import argparse
import glob
import json
import os
import re
import sys
import time

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
DEFAULT_ANNO = f"{ROOT}/hawk_anomaly/Annotation/All_Mix/all_videos_test.local.json"
DEFAULT_VIDEOS = f"{ROOT}/hawk_anomaly/Videos"


# ---------------------------------------------------------------------------
# 지표
# ---------------------------------------------------------------------------
# 정답 설명 중 일부는 서술이 아니라 **거부 응답**이다 — "As an AI developed by OpenAI,
# I cannot...". 원본 데이터셋의 캡션이 LLM 으로 생성되면서 섞여 들어간 것으로, UCF-Crime
# 에서 200/1854 = 10.8% 를 차지한다(다른 데이터셋에는 없다).
#
# 이런 클립을 평가에 두면 생성문을 거부문과 대조하게 되어 BLEU 가 의미를 잃고, 장면 어휘
# 재현율도 정답 쪽에 잴 어휘가 없어 표본에서 빠진다. 판정자 기반 지표는 더 나쁘다 — 거부문을
# 기준으로 "합리성"을 매기게 된다. 따라서 평가에서 제외하고 제외 건수를 결과에 기록한다.
#
# 학습 데이터는 건드리지 않는다. 원본과 동일한 데이터로 학습한다는 통제가 이 논문의 비교
# 논리이므로, 평가 시점에만 걸러낸다.
REFUSAL_PAT = re.compile(
    r"(as an ai|i am unable|i cannot|without having access|i don't have access|"
    r"unable to (view|provide|analyz)|developed by openai|no capability to view|"
    r"i'm sorry|cannot provide)", re.I)


def is_refusal(text):
    return bool(text) and bool(REFUSAL_PAT.search(text))


def filter_refusals(records, field_gt):
    """거부 응답을 정답으로 가진 레코드를 걸러내고 (남은 것, 제외 수) 를 돌려준다."""
    keep = [r for r in records if not is_refusal(r.get(field_gt))]
    return keep, len(records) - len(keep)


def compute_bleu(references, hypotheses):
    """BLEU-1..4. 캡셔닝 논문 표준 구현(pycocoevalcap)을 쓴다.

    자체 구현을 쓰면 토큰화 차이 때문에 공개 수치와 비교할 수 없게 된다.
    """
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer  # noqa: F401  (문서화 목적)

    gts = {str(i): [r.strip().lower()] for i, r in enumerate(references)}
    res = {str(i): [h.strip().lower()] for i, h in enumerate(hypotheses)}
    scores, _ = Bleu(4).compute_score(gts, res)
    return {f"BLEU-{i + 1}": float(s) for i, s in enumerate(scores)}


JUDGE_PROMPT = """You are grading a model-generated description of a video against a \
reference description written by a human annotator.

Reference (ground truth):
{reference}

Model output:
{hypothesis}

Score the model output on three dimensions, each from 0.0 to 1.0:

- reasonability: is the described event logically coherent and plausible given the \
reference? Penalise contradictions and hallucinated causes.
- detail: does it capture the specific content of the reference (objects, place, \
conditions, what happened) rather than generic filler?
- consistency: is the output internally consistent and consistent with the reference, \
without self-contradiction or drift?

Respond with JSON only, no prose:
{{"reasonability": <float>, "detail": <float>, "consistency": <float>}}"""


# 평가 전용 장면 어휘 범주. 학습의 추출기(구문 역할 기반)와 **독립적으로** 정의한다.
#
# 왜 분리하는가. 학습은 `extract_background_entities_sentence`(의존 구문 역할이 주어·목적어가
# 아닌 명사 + 형용사)로 정적 스트림의 감독 목표를 만든다. 평가가 같은 추출기를 쓰면, 그 집합을
# 생성하도록 학습된 모델이 그 집합의 재현율에서 높은 점수를 받는 것이 당연해진다 — 이해의
# 증거가 아니라 **학습 충실도의 증거**가 되어 순환한다.
#
# 그래서 평가는 "장면을 규정하는 조건"이라는 의미 범주를 명시적 어휘로 고정한다. 이 목록은
# 결과를 보기 전에 확정했으며, 모델 출력이 아니라 이 연구가 주장하는 대상(노면 상태·기상·
# 조명·장소 유형)에서 도출했다.
SCENE_LEXICON = {
    # 노면·표면 상태
    "wet", "icy", "snowy", "slippery", "muddy", "dry", "flooded", "gravel", "pavement",
    "asphalt", "sidewalk", "pothole", "curb", "surface", "road", "lane", "crosswalk",
    # 기상·시간·조명
    "rain", "rainy", "snow", "fog", "foggy", "storm", "stormy", "windy", "night",
    "dark", "dim", "bright", "sunny", "cloudy", "daylight", "dusk", "shadow", "lit",
    "illuminated", "lighting", "streetlight",
    # 장소 유형
    "highway", "freeway", "street", "intersection", "tunnel", "bridge", "parking",
    "garage", "alley", "corridor", "hallway", "lobby", "store", "shop", "mall",
    "station", "platform", "stairs", "staircase", "escalator", "elevator", "entrance",
    "campus", "park", "playground", "field", "yard", "warehouse", "factory", "office",
    "room", "kitchen", "bank", "market", "restaurant", "school", "hospital",
    # 장면 구조물
    "fence", "wall", "barrier", "railing", "guardrail", "pole", "sign", "signal",
    "traffic", "building", "roof", "window", "door", "gate", "bench", "tree",
    # 밀집도·상태
    "crowded", "empty", "busy", "deserted", "narrow", "wide", "steep",
}


def scene_word_recall(references, hypotheses, lexicon=None):
    """장면 어휘 재현율 — 이 연구의 주장을 직접 재는 지표.

    BLEU 는 문장 전체를 보므로 배경 서술이 좋아져도 다른 내용에 희석되고, BSI 는 배경에
    *민감한가*를 잴 뿐 *정확한가*를 재지 않는다. 정작 이 연구가 주장하는 것은
    **"기존 모델이 놓치던 정적 장면 맥락을 우리 모델은 말한다"** 이므로, 그것을 직접
    재는 지표가 필요하다.

    정답 캡션에 나타난 장면 어휘를 **고정 어휘집**(`SCENE_LEXICON`)으로 식별하고, 모델
    출력이 그중 몇 개를 재현했는지 센다. 어휘집은 학습의 추출기와 독립이며, 그 이유는
    `SCENE_LEXICON` 주석에 적었다.

    함께 보고하는 값:
      recall     정답의 장면 어휘 중 맞힌 비율        ← 주 지표
      precision  말한 장면 어휘 중 정답에 있던 비율    ← 남발 방지
      n_ref      정답에 장면 어휘가 있던 클립 수      ← 표본 수 공개

    precision 이 필요한 이유는 합성 검증에서 드러난다. 장면 어휘를 무차별로 나열하는 응답은
    recall 0.727 을 얻지만 precision 0.381 로 걸러진다(정확히 서술한 응답은 1.000 / 1.000).
    recall 만 보고하면 "장면 단어를 많이 말하는" 전략이 이긴다.
    """
    lex = lexicon if lexicon is not None else SCENE_LEXICON

    def scene_terms(text):
        toks = {w.strip(".,;:!?\"'()").lower() for w in text.split()}
        return toks & lex

    hits = misses = spurious = 0
    n_ref = 0
    for ref, hyp in zip(references, hypotheses):
        gold = scene_terms(ref)
        if not gold:              # 정답에 장면 어휘가 없으면 잴 것이 없다
            continue
        n_ref += 1
        said = scene_terms(hyp)
        hit = gold & said
        hits += len(hit)
        misses += len(gold - said)
        spurious += len(said - gold)   # 정답에 없는 장면 어휘를 말한 것

    if n_ref == 0:
        return {"recall": float("nan"), "n_ref": 0}
    denom_p = hits + spurious
    return {
        "recall": hits / (hits + misses),
        "precision": hits / denom_p if denom_p else float("nan"),
        "n_ref": n_ref,
        "n_gold_words": hits + misses,
        "lexicon_size": len(lex),
        # 학습 추출기와 독립임을 결과 파일에 남긴다 — 순환성 지적에 대한 답이다.
        "extractor": "fixed scene lexicon (independent of the L_BL training extractor)",
    }


def judge_gpt_guided(records, judge_model, field_gt, field_pred, limit=None, verbose=True):
    """GPT-guided 3종을 외부 LLM 판정자로 매긴다.

    ⚠️ 비교 가능성에 대한 경고 — 논문에 쓸 때 반드시 지킬 것.
    원본 HAWK 의 Reasonability/Detail/Consistency 는 **GPT 가 매긴 점수**다. 판정자가
    다르면 절대값이 달라지므로, 여기서 나온 점수를 그 표와 나란히 놓아서는 안 된다.
    이 지표는 **같은 판정자로 매긴 arm 끼리의 비교**에만 쓰고, 논문 대 논문 비교는
    판정자가 없는 BLEU 가 담당한다.

    심사 통과를 위해 함께 보고해야 하는 것:
      - 판정 모델의 정확한 버전(결과 파일에 기록됨)과 프롬프트 전문(부록)
      - 제시 순서 무작위화 여부 — 본 구현은 참조/가설을 고정 순서로 제시하므로,
        arm 간 비교에서는 편향이 대칭이나 절대값 해석에는 주의
      - 최소 100 샘플에 대한 인간-판정자 일치도 (별도 측정 필요)
    """
    import json as _json
    import os as _os
    import re as _re

    from google import genai
    from google.genai import types

    key = _os.environ.get("GEMINI_API_KEY")
    if not key:
        token_path = f"{ROOT}/.gemini_token"
        if _os.path.exists(token_path):
            key = open(token_path).read().strip()
    if not key:
        raise RuntimeError(
            "판정자 API 키가 없습니다. GEMINI_API_KEY 를 설정하거나 "
            f"{ROOT}/.gemini_token 을 두십시오."
        )

    client = genai.Client(api_key=key)
    subset = records[:limit] if limit else records
    dims = ("reasonability", "detail", "consistency")
    totals = {d: [] for d in dims}
    failures = 0

    for i, rec in enumerate(subset, 1):
        gt, pred = rec.get(field_gt) or "", rec.get(field_pred) or ""
        if not gt or not pred:
            continue
        try:
            resp = client.models.generate_content(
                model=judge_model,
                contents=JUDGE_PROMPT.format(reference=gt, hypothesis=pred),
                # temperature 0 — 판정자는 결정론이어야 재현 가능하다.
                config=types.GenerateContentConfig(temperature=0.0),
            )
            text = resp.text.strip()
            match = _re.search(r"\{.*\}", text, _re.S)
            scores = _json.loads(match.group(0) if match else text)
            for d in dims:
                totals[d].append(float(scores[d]))
            rec["judge_scores"] = {d: float(scores[d]) for d in dims}
        except Exception as exc:
            failures += 1
            if failures <= 3:
                print(f"  [판정 실패] {rec.get('video')}: {exc}")

        if verbose and i % 50 == 0:
            print(f"  판정 {i}/{len(subset)} (실패 {failures})")

    if not totals["reasonability"]:
        raise RuntimeError(f"판정에 모두 실패했습니다 (시도 {len(subset)}건).")

    out = {d: sum(v) / len(v) for d, v in totals.items()}
    out["_judge_model"] = judge_model
    out["_n_judged"] = len(totals["reasonability"])
    out["_n_failed"] = failures
    out["_comparable_to_published"] = False   # 판정자가 다르므로 항상 False
    return out


# ---------------------------------------------------------------------------
# 추론
# ---------------------------------------------------------------------------
def build_chat(cfg_path, ckpt, gpu_id):
    from hawk.common.config import Config
    from hawk.common.registry import registry
    from hawk.conversation.conversation_video import Chat
    import hawk.models  # noqa: F401
    import hawk.processors  # noqa: F401
    import hawk.tasks  # noqa: F401

    class _Args:
        cfg_path = None
        options = None

    args = _Args()
    args.cfg_path = cfg_path
    cfg = Config(args)

    model_cfg = cfg.model_cfg
    model_cfg.device_8bit = gpu_id
    if ckpt:
        model_cfg.ckpt = ckpt          # arm 별 체크포인트로 덮어쓴다

    model = registry.get_model_class(model_cfg.arch).from_config(model_cfg).to(f"cuda:{gpu_id}")
    model.eval()

    # 평가에는 반드시 eval processor(`alpro_video_eval`)를 써야 한다. train processor 는
    # RandomResizedCrop 이 걸려 있어 같은 클립을 두 번 평가해도 결과가 달라진다.
    # eval.yaml 은 `train:` 키 아래에 eval processor 를 두고 있으므로 둘 다 받아들이되,
    # 최종적으로 선택된 프로세서 이름을 검사해 train 계열이면 거부한다.
    vp_all = cfg.datasets_cfg.webvid.vis_processor
    vp_cfg = vp_all.get("eval", None) or vp_all.get("train")
    if "eval" not in vp_cfg.name:
        raise ValueError(
            f"평가에 train processor('{vp_cfg.name}')가 선택되었습니다. "
            "RandomResizedCrop 때문에 결과가 재현되지 않습니다. "
            "config 의 vis_processor 를 'alpro_video_eval' 로 바꾸십시오."
        )
    vis_processor = registry.get_processor_class(vp_cfg.name).from_config(vp_cfg)

    return Chat(model, vis_processor, device=f"cuda:{gpu_id}")


def generate_one(chat, video_path, question, max_new_tokens, num_beams):
    from hawk.conversation.conversation_video import conv_llava_llama_2

    conv = conv_llava_llama_2.copy()
    conv.system = ""
    img_list = []
    chat.upload_video_without_audio(video_path, conv, img_list)
    chat.ask(question, conv)
    text, _ = chat.answer(
        conv=conv,
        img_list=img_list,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        do_sample=False,            # 결정론 — 설계 원칙 1
    )
    return text.strip()


# ---------------------------------------------------------------------------
def run_eval(args):
    with open(args.anno) as f:
        samples = json.load(f)
    if args.limit:
        samples = samples[: args.limit]

    print(f"[eval] 클립 {len(samples)}개 | ckpt={args.ckpt or '(config 기본값)'}")
    chat = build_chat(args.cfg, args.ckpt, args.gpu_id)

    records, failures = [], 0
    t0 = time.time()
    for i, sample in enumerate(samples, 1):
        video_path = os.path.join(args.videos_dir, sample["video"])
        try:
            description = generate_one(
                chat, video_path, args.describe_prompt, args.max_new_tokens, args.num_beams
            )
            qa = sample.get("QA") or []
            answer = (
                generate_one(chat, video_path, qa[0]["q"], args.max_new_tokens, args.num_beams)
                if qa else None
            )
            records.append({
                "video": sample["video"],
                "gt_description": sample.get("description", ""),
                "pred_description": description,
                "question": qa[0]["q"] if qa else None,
                "gt_answer": qa[0]["a"] if qa else None,
                "pred_answer": answer,
            })
        except Exception as exc:
            failures += 1
            print(f"  [실패] {sample['video']}: {exc}")

        if i % 25 == 0:
            rate = i / (time.time() - t0)
            print(f"  {i}/{len(samples)}  ({rate:.2f} clip/s, 실패 {failures})")

    gts = [r["gt_description"] for r in records]
    preds = [r["pred_description"] for r in records]

    metrics = {
        "description": compute_bleu(gts, preds),
        # 연구 주장을 직접 재는 지표 — 배경 서술 능력. BLEU 와 별개로 보고한다.
        "scene_word": scene_word_recall(gts, preds),
    }
    qa_records = [r for r in records if r["gt_answer"] and r["pred_answer"]]
    if qa_records:
        metrics["qa"] = compute_bleu(
            [r["gt_answer"] for r in qa_records], [r["pred_answer"] for r in qa_records]
        )

    if args.judge:
        print(f"[eval] 판정자 실행: {args.judge}")
        metrics["judge_description"] = judge_gpt_guided(
            records, args.judge, "gt_description", "pred_description", args.judge_limit
        )
        if qa_records:
            metrics["judge_qa"] = judge_gpt_guided(
                qa_records, args.judge, "gt_answer", "pred_answer", args.judge_limit
            )

    out = {
        # 재현에 필요한 조건은 전부 결과 파일에 박아 둔다. 이 값이 다른 두 파일을
        # 나란히 놓고 비교해서는 안 된다.
        "config": {
            "cfg": args.cfg,
            "ckpt": args.ckpt,
            "anno": args.anno,
            "n_clips": len(records),
            "n_failed": failures,
            "decoding": {"do_sample": False, "num_beams": args.num_beams,
                         "max_new_tokens": args.max_new_tokens},
            "describe_prompt": args.describe_prompt,
            "judge": args.judge,
        },
        "metrics": metrics,
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\n[eval] 완료 — {args.out}")
    for task, scores in metrics.items():
        print(f"  {task}: " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    return 0


def run_compare(paths):
    """여러 arm 의 결과를 나란히 놓는다. 비교 불가 조건은 경고한다."""
    loaded = []
    for p in sorted(paths):
        with open(p) as f:
            loaded.append((os.path.basename(p), json.load(f)))
    if not loaded:
        print("비교할 결과 파일이 없습니다.")
        return 1

    base = loaded[0][1]["config"]
    for name, d in loaded[1:]:
        for key in ("anno", "decoding", "describe_prompt"):
            if d["config"][key] != base[key]:
                print(f"  ⚠️ {name}: '{key}' 가 기준과 다릅니다 — 직접 비교 불가")

    keys = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"]
    print(f"\n{'arm':<28}{'n':>6}" + "".join(f"{k:>10}" for k in keys))
    for name, d in loaded:
        m = d["metrics"].get("description", {})
        print(f"{name:<28}{d['config']['n_clips']:>6}" +
              "".join(f"{m.get(k, float('nan')):>10.4f}" for k in keys))
    print("\n주의: 단일 실행 점추정입니다. seed 반복 없이 arm 간 소수점 차이를 "
          "주장하지 마십시오.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval_configs/eval.yaml")
    ap.add_argument("--ckpt", default=None, help="arm 별 체크포인트 (config 값을 덮어씀)")
    ap.add_argument("--anno", default=DEFAULT_ANNO)
    ap.add_argument("--videos-dir", default=DEFAULT_VIDEOS)
    ap.add_argument("--out", default="experiments/out/eval.json")
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--describe-prompt",
                    default="Describe the video and identify whether anything anomalous "
                            "happens, including where it happens and why it is dangerous.")
    ap.add_argument("--judge", default=None,
                    help="판정자 모델 id (예: gemini-2.5-flash). 미지정 시 BLEU 만 계산. "
                         "판정 점수는 같은 판정자로 매긴 arm 끼리만 비교 가능하다")
    ap.add_argument("--judge-limit", type=int, default=0,
                    help="판정할 샘플 수 상한 (0이면 전체)")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="결과 JSON 들을 비교 출력하고 종료")
    args = ap.parse_args()

    if args.compare is not None:
        paths = args.compare or glob.glob("experiments/out/eval_*.json")
        return run_compare(paths)
    return run_eval(args)


if __name__ == "__main__":
    sys.exit(main())
