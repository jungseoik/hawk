#!/usr/bin/env python
"""Places365 장면 인식 프로브 — 외부·객관 라벨로 배경 이해를 잰다.

왜 필요한가
-----------
지금까지 배경 이해의 근거가 전부 **우리가 만든** Scene-word Recall 이었다.
"당신들 지표에서만 좋은 것 아니냐"에 답하려면 우리가 만들지 않은 라벨이 필요하다.
Places365 는 365 개 장면 범주에 객관적 라벨이 붙어 있고 우리 학습에 쓰인 적이 없다.

설계
----
정지 이미지를 32 프레임으로 복제해 모델에 넣는다. 그러면 광학 흐름이 0 이 되어
  마스크 M = 0  →  움직임 스트림 = 0,  정적 스트림 = 화면 전체
즉 **배경 분기만 장면 정보를 받는 조건**이 된다. 배경 분기가 장면을 담고 있다면
`flow` arm 이 여기서 유리해야 한다. 배경 분기가 상수로 붕괴해 있다면 차이가 없을 것이다.

⚠ 이 조건은 모든 모델에 대해 **분포 밖(out-of-distribution)** 이다. 비디오로 학습한
모델에 정지 이미지를 넣기 때문이다. 조건이 모든 모델에 동일하므로 arm 간 비교는
유효하지만, 절대값을 비디오 성능으로 해석하면 안 된다. 논문에 쓸 때 명시할 것.
"""
import argparse, json, os, sys
import numpy as np, torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import importlib.util as _u
_s = _u.spec_from_file_location("_ev", os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate.py"))
_ev = _u.module_from_spec(_s)
try: _s.loader.exec_module(_ev)
except SystemExit: pass

PROMPT = ("Describe the place shown in this video: what kind of location is it, "
          "and what does the surrounding environment look like?")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/eval_configs/eval.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    a = ap.parse_args()

    _ev.enforce_determinism(0)
    ub = _ev.detect_use_background(a.ckpt)
    chat = _ev.build_chat(a.cfg, a.ckpt, a.gpu_id, use_background=ub)
    print(f"[probe] use_background={ub}", flush=True)
    dev = f"cuda:{a.gpu_id}"

    meta = json.load(open(a.subset))
    if a.limit: meta = meta[:a.limit]
    recs = []
    for i, m in enumerate(meta, 1):
        try:
            img = Image.open(m["file"]).convert("RGB").resize((224, 224))
            arr = np.asarray(img)                                   # H,W,C
            clip = np.repeat(arr[None, ...], 32, axis=0)            # T,H,W,C
            t = torch.from_numpy(clip).permute(3, 0, 1, 2).float()  # C,T,H,W
            t = chat.vis_processor.transform(t)
            t = t.unsqueeze(0).to(dev)
            with torch.no_grad():
                ea, _, _ = chat.model.encode_videoQformer_visual(t)
                em, _, _ = chat.model.encode_videoQformer_visual(torch.zeros_like(t), motion=True)
                streams = [ea, em]
                if getattr(chat.model, "use_background", True):
                    eb, _, _ = chat.model.encode_videoQformer_visual(t, background=True)
                    streams.append(eb)
                img_list = [torch.cat(tuple(streams), dim=1)]
            from hawk.conversation.conversation_video import conv_llava_llama_2
            conv = conv_llava_llama_2.copy()
            conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + PROMPT)
            conv.append_message(conv.roles[1], None)
            out = chat.answer(conv=conv, img_list=img_list, max_new_tokens=a.max_new_tokens,
                              num_beams=1)[0]
        except Exception as e:
            out = ""
            if i <= 3: print(f"  [실패] {m['id']}: {e}", flush=True)
        recs.append({**m, "pred": out})
        if i % 50 == 0: print(f"  {i}/{len(meta)}", flush=True)

    json.dump({"ckpt": a.ckpt, "use_background": ub, "prompt": PROMPT, "records": recs},
              open(a.out, "w"), ensure_ascii=False)
    ok = sum(1 for r in recs if r["pred"].strip())
    print(f"  저장 {a.out} · 생성 성공 {ok}/{len(recs)}")


if __name__ == "__main__":
    main()
