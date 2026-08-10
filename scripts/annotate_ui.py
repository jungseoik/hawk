#!/usr/bin/env python
"""Background-critical 라벨링 UI — 키보드로 빠르게, 블라인드로.

왜 이런 순서인가
----------------
라벨은 "배경을 가리면 이상을 알 수 있는가"를 묻는다. 사람과 LLM 판정자가 **같은 과제**를
수행해야 둘의 일치도(κ)가 의미를 갖는다. 그래서 두 판정자 모두 아래 순서를 따른다.

  1단계  동적 프레임만 (배경이 검게 지워진 M⊙x) → "이상이 보이나?"
  2단계  원본 프레임                              → "이상이 보이나?"
  라벨   1에서 못 보고 2에서 보임 → context_critical
         둘 다 보이나 2에서 더 구체적            → context_dependent
         1에서 이미 충분                          → motion_sufficient

**정답 설명은 판정을 마친 뒤에만 공개한다.** 먼저 보여주면 답을 알고 판정하게 되어
일치도 측정이 무의미해진다. 공개 단계에서는 영문 원문과 한국어 번역을 함께 보여준다 —
판정은 영상으로 하고, 확인만 편한 언어로 하기 위해서다.

또한 이 화면은 **어떤 모델의 출력도, LLM 판정자의 라벨도 보여주지 않는다.** 저자가
"우리 모델이 이기는 클립"을 고르지 않았음을 절차로 보이기 위한 블라인딩이다.

사용법
------
    $CERBERUS_PY scripts/annotate_ui.py --port 7860
    # 브라우저에서 http://localhost:7860  (원격이면 SSH 포트포워딩)

키보드
    1 / 2 / 3   해당 질문에 답 (자세히 / 대략 / 못 알아봄)
    ←           이전 클립으로
    S           건너뛰기 (판단 보류)
    저장은 매 판정마다 자동 (--out JSON)
"""
import argparse
import base64
import io
import json
import os
import random
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
QUEUE = f"{ROOT}/hawk/experiments/bg_critical_benchmark/annotation_queue.json"
VIDEOS = f"{ROOT}/hawk_anomaly/Videos"
ANNO = f"{ROOT}/hawk_anomaly/Annotation/All_Mix/all_videos_all.local.json"
MAG_THRESHOLD = 0.2
N_FRAMES = 6

_state = {"items": [], "idx": 0, "labels": {}, "out": None, "desc": {}, "cache": {}}
_lock = threading.Lock()


# ---------------------------------------------------------------------------
def _frames(video_path, n=N_FRAMES):
    """원본 프레임과 동적 프레임(배경 제거)을 학습과 동일한 방식으로 만든다."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        return [], []
    idxs = np.linspace(0, total - 1, n).astype(int)
    originals, motions = [], []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(i) - 1, 0))
        ok_p, prev = cap.read()
        ok_c, cur = cap.read()
        if not (ok_p and ok_c):
            continue
        prev_s = cv2.resize(prev, (448, 448))
        cur_s = cv2.resize(cur, (448, 448))
        flow = cv2.calcOpticalFlowFarneback(
            cv2.cvtColor(prev_s, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(cur_s, cv2.COLOR_BGR2GRAY),
            None, 0.5, 3, 10, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mask = (mag > MAG_THRESHOLD).astype(np.uint8)[..., None]
        originals.append(cur_s)
        motions.append(cur_s * mask)          # 배경이 검게 지워진 동적 스트림
    cap.release()
    return originals, motions


def _b64(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return base64.b64encode(buf).decode() if ok else ""


def _payload(idx):
    """한 클립의 프레임을 만들어 돌려준다 (직전/다음 것을 미리 캐시)."""
    item = _state["items"][idx]
    key = item["video_path"]
    if key not in _state["cache"]:
        o, m = _frames(os.path.join(VIDEOS, key))
        _state["cache"][key] = ([_b64(x) for x in o], [_b64(x) for x in m])
        if len(_state["cache"]) > 12:                    # 메모리 상한
            _state["cache"].pop(next(iter(_state["cache"])))
    originals, motions = _state["cache"][key]
    d = _state["desc"].get(key, {})
    return {
        "idx": idx, "total": len(_state["items"]),
        "clip_id": item["clip_id"], "dataset": item["source_dataset"],
        "motion_ratio": item.get("motion_ratio"),
        "originals": originals, "motions": motions,
        # 설명은 판정 후에만 화면에 노출된다 (프런트에서 제어)
        "desc_en": d.get("en", ""), "desc_ko": d.get("ko", ""),
        "done": len(_state["labels"]),
        "existing": _state["labels"].get(item["clip_id"]),
    }


def _save():
    with open(_state["out"], "w") as f:
        json.dump(_state["labels"], f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
PAGE = """<!doctype html><meta charset="utf-8">
<title>Background-critical 라벨링</title>
<style>
 :root{--bg:#111;--fg:#eee;--dim:#888;--acc:#4ea1ff;--warn:#e8a33d}
 body{background:var(--bg);color:var(--fg);font-family:system-ui,-apple-system,sans-serif;
      margin:0;padding:16px 20px}
 .top{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:10px}
 .prog{color:var(--dim);font-size:13px}
 h2{margin:0 0 4px;font-size:17px}
 .q{color:var(--acc);font-size:15px;margin:10px 0 6px}
 .strip{display:flex;gap:6px;flex-wrap:wrap}
 .strip img{width:180px;height:180px;object-fit:cover;border-radius:6px;background:#000}
 .keys{margin-top:12px;font-size:14px;line-height:1.9}
 .k{display:inline-block;min-width:22px;padding:2px 7px;margin-right:8px;border-radius:4px;
    background:#2a2a2a;border:1px solid #444;font-weight:600}
 .reveal{margin-top:14px;padding:12px 14px;background:#1b1b1b;border-radius:8px;
         border-left:3px solid var(--warn)}
 .reveal .en{color:var(--dim);font-size:13px;margin-bottom:6px}
 .reveal .ko{font-size:15px;line-height:1.6}
 .hide{display:none}
 .meta{color:var(--dim);font-size:12px;margin-top:6px}
</style>
<div class="top">
  <h2 id="cid">…</h2>
  <div class="prog"><span id="pos"></span> · 완료 <span id="done"></span></div>
</div>
<div class="q" id="q">불러오는 중…</div>
<div class="strip" id="strip"></div>
<div class="meta" id="meta"></div>
<div class="keys" id="keys"></div>
<div class="reveal hide" id="rev">
  <div class="en" id="ren"></div><div class="ko" id="rko"></div>
</div>
<script>
let D=null, stage=0, a1=null;
const $=id=>document.getElementById(id);

async function load(i){
  const r = await fetch('/api/item'+(i!==undefined?'?i='+i:''));
  D = await r.json(); stage=0; a1=null; render();
}
function render(){
  $('cid').textContent = D.clip_id.slice(0,60);
  $('pos').textContent = (D.idx+1)+' / '+D.total;
  $('done').textContent = D.done;
  $('meta').textContent = D.dataset + '  ·  motion ratio ' +
      (D.motion_ratio!=null? D.motion_ratio.toFixed(3):'?');
  $('rev').classList.add('hide');
  const imgs = (stage===0? D.motions : D.originals);
  $('strip').innerHTML = imgs.map(b=>'<img src="data:image/jpeg;base64,'+b+'">').join('');
  if(stage===0){
    $('q').textContent = '1단계 — 배경을 지운 화면입니다. 여기서 이상을 알아볼 수 있습니까?';
    $('keys').innerHTML = '<span class="k">1</span>충분히 알겠다  '
      + '<span class="k">2</span>뭔가 있는 것 같은데 불충분  '
      + '<span class="k">3</span>전혀 모르겠다<br>'
      + '<span class="k">←</span>이전  <span class="k">S</span>보류';
  } else {
    $('q').textContent = '2단계 — 원본 화면입니다. 여기서는 이상을 알아볼 수 있습니까?';
    $('keys').innerHTML = '<span class="k">1</span>충분히 알겠다  '
      + '<span class="k">2</span>대략 알겠다  '
      + '<span class="k">3</span>모르겠다<br>'
      + '<span class="k">←</span>1단계로';
  }
}
function label(a1,a2){
  // 배경 없이 못 알아봤는데 원본에서 알아봄 → 배경이 결정적
  if(a1>=3 && a2<=2) return 'context_critical';
  if(a1===2 && a2===1) return 'context_dependent';
  if(a1===1) return 'motion_sufficient';
  return 'context_dependent';
}
async function send(lab, skip){
  await fetch('/api/label',{method:'POST',headers:{'Content-Type':'application/json'},
     body:JSON.stringify({clip_id:D.clip_id, label:lab, skip:!!skip,
                          a_motion:a1, video_path:D.clip_id})});
  if(!skip){
    $('ren').textContent = D.desc_en || '(설명 없음)';
    $('rko').textContent = D.desc_ko || '';
    $('rev').classList.remove('hide');
    $('q').textContent = '판정: ' + lab + '  —  아무 키나 누르면 다음';
    stage = 2;
  } else { load(D.idx+1); }
}
document.addEventListener('keydown', async e=>{
  if(!D) return;
  if(stage===2){ load(D.idx+1); return; }
  const k = e.key.toLowerCase();
  if(k==='arrowleft'){ if(stage===1){stage=0;render();} else load(Math.max(D.idx-1,0)); return; }
  if(k==='s' && stage===0){ send(null,true); return; }
  if(['1','2','3'].includes(k)){
    if(stage===0){ a1=parseInt(k); stage=1; render(); }
    else { send(label(a1,parseInt(k)), false); }
  }
});
load();
</script>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/item"):
            with _lock:
                if "?i=" in self.path:
                    i = int(self.path.split("?i=")[1])
                    _state["idx"] = max(0, min(i, len(_state["items"]) - 1))
                self._json(_payload(_state["idx"]))
            return
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        with _lock:
            if not data.get("skip") and data.get("label"):
                _state["labels"][data["clip_id"]] = {
                    "motion_sufficiency": data["label"],
                    "a_motion_only": data.get("a_motion"),
                    "annotator": _state["annotator"],
                }
                _save()
            _state["idx"] = min(_state["idx"] + 1, len(_state["items"]) - 1)
        self._json({"ok": True})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", default=QUEUE)
    ap.add_argument("--out", default=f"{ROOT}/hawk/experiments/bg_critical_benchmark/labels_human.json")
    ap.add_argument("--ko", default=f"{ROOT}/hawk/experiments/bg_critical_benchmark/descriptions_ko.json",
                    help="한국어 번역 (판정 후 확인용). 없으면 영문만 표시")
    ap.add_argument("--sample", type=int, default=150,
                    help="무작위 표본 크기 (일치도 측정용). 0이면 전체")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--annotator", default="human_1")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    with open(args.queue) as f:
        items = json.load(f)
    if args.sample:
        items = random.Random(args.seed).sample(items, min(args.sample, len(items)))

    # 정답 설명(영문)과 한국어 번역을 미리 붙여 둔다. 판정 *후에만* 화면에 나온다.
    desc = {}
    if os.path.exists(ANNO):
        with open(ANNO) as f:
            for s in json.load(f):
                desc[s["video"]] = {"en": s.get("description", "")}
    # 번역본은 레포 안에 고정 위치로 둔다. --out(라벨 저장 위치)을 따라가게 하면
    # 저장 경로를 바꾸는 순간 번역이 사라진다.
    ko_path = args.ko
    if os.path.exists(ko_path):
        with open(ko_path) as f:
            for k, v in json.load(f).items():
                desc.setdefault(k, {})["ko"] = v
    else:
        print(f"[안내] 한국어 번역이 없습니다. scripts/translate_descriptions.py 로 만들면 "
              f"영문과 함께 표시됩니다: {ko_path}")

    _state.update(items=items, out=args.out, desc=desc, annotator=args.annotator)
    if os.path.exists(args.out):
        with open(args.out) as f:
            _state["labels"] = json.load(f)
        print(f"[재개] 기존 라벨 {len(_state['labels'])}건을 이어서 진행합니다")

    print(f"클립 {len(items)}건 · 저장 → {args.out}")
    print(f"브라우저에서 http://localhost:{args.port}  (원격이면 SSH 포트포워딩)")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
