#!/usr/bin/env python
"""장면-인과 판정 검증 UI — 정답 설명을 읽고 4택.

무엇을 판정하나
---------------
"이 정답 설명에서 **정적 장면 조건이 원인인가, 배경일 뿐인가.**"

기존 UI(마스킹 영상 → 원본 영상 2단계)와 다른 점:

  기존   전경이 지워진 저정보 영상을 보고 "모션만으로 판별 가능한가"를 판단
         → 사람과 LLM 이 서로 다른 정보를 봐서 κ 가 0.383 에 머물렀다
  현재   정답 설명 텍스트를 읽고 "장면 조건이 원인인가"를 판단
         → 두 판정자가 같은 것을 보고, 채점 기준(정답 설명과의 비교)과도 일치한다

이 UI 의 목적은 **전량 라벨링이 아니라 LLM 판정의 신뢰도 검증**이다. 80 건만 채우면
Cohen's κ 를 계산할 수 있고, 나머지 391 건은 LLM 판정을 쓴다.

블라인드
--------
LLM 판정은 사람이 답을 낸 **뒤에만** 보인다(`--reveal` 로 끄면 아예 안 보인다).
먼저 보이면 일치도가 사람의 독립 판단이 아니라 동조를 재게 된다.

키보드
------
  1  incidental   장면이 나오지만 원인은 행위·객체에 있다 (장소를 바꿔도 설명이 같다)
  2  causal       정적 장면 조건이 원인·위험도의 일부다 (그 조건을 빼면 설명이 달라진다)
  3  no_scene     정적 장면 조건이 설명에 아예 없다
  4  normal       이상 사건 자체가 없다
  ←  이전으로     s 건너뛰기     f 프레임 접기/펼치기

사용
----
    $CERBERUS_PY scripts/validate_ui.py            # http://localhost:7861
    $CERBERUS_PY scripts/validate_ui.py --port 7862 --no-reveal
"""
import argparse
import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
BENCH = f"{ROOT}/hawk/experiments/bg_critical_benchmark"
SAMPLE = f"{BENCH}/validation_sample.json"
KO = f"{BENCH}/descriptions_ko_validation.json"
LLM = f"{BENCH}/labels_scene_causal_llm.json"
OUT = f"{BENCH}/labels_scene_causal_human.json"
VIDEOS = f"{ROOT}/hawk_anomaly/Videos"

LABELS = ["incidental", "causal", "no_scene", "normal"]
N_FRAMES = 4

_lock = threading.Lock()
_state = {"items": [], "ko": {}, "llm": {}, "labels": {}, "out": OUT, "mode": "audit"}


def _frames(video_path, n=N_FRAMES):
    """참고용 프레임. 판정은 텍스트로 하되, 애매할 때 눈으로 확인할 수 있게 둔다."""
    try:
        import decord
        from PIL import Image
        vr = decord.VideoReader(os.path.join(VIDEOS, video_path), num_threads=1)
        total = len(vr)
        idx = [int(total * (i + 0.5) / n) for i in range(n)]
        out = []
        for i in idx:
            arr = vr[i].asnumpy()
            img = Image.fromarray(arr)
            img.thumbnail((420, 420))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            out.append("data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode())
        return out
    except Exception as exc:
        return [f"__error__{exc}"]


def _payload(idx):
    items = _state["items"]
    idx = max(0, min(idx, len(items) - 1))
    it = items[idx]
    vid = it["video_path"]
    done = len(_state["labels"])
    return {
        "idx": idx,
        "total": len(items),
        "done": done,
        "clip_id": it["clip_id"],
        "dataset": it["source_dataset"],
        "en": it["description"],
        "ko": _state["ko"].get(vid, "(번역 없음)"),
        "frames": _frames(vid),
        "prev": _state["labels"].get(it["clip_id"], {}).get("label"),
        "mode": _state["mode"],
        # blind 모드에서는 LLM 판정을 여기 싣지 않는다. 화면에 감추더라도 응답에 들어
        # 있으면 브라우저에 이미 도착한 것이고, 블라인드는 표시 여부가 아니라 정보 도달
        # 여부로 정의된다. audit 모드에서는 의도적으로 미리 보여준다 — 목적이 독립
        # 판정이 아니라 LLM 라벨의 검수이기 때문이다(κ는 이 모드에서 유효하지 않다).
        "llm": _state["llm"].get(it["clip_id"]) if _state["mode"] == "audit" else None,
    }


def _save():
    with _lock:
        with open(_state["out"], "w") as f:
            json.dump(list(_state["labels"].values()), f, ensure_ascii=False, indent=2)


PAGE = """<!doctype html><meta charset=utf-8><title>장면-인과 판정 검증</title>
<style>
 :root{--bg:#0f1115;--fg:#e8eaed;--dim:#9aa0a6;--card:#1a1d23;--line:#2a2f38;
       --a:#4f8cff;--b:#ff8c4f;--c:#8c8c8c;--d:#5fbf7f}
 *{box-sizing:border-box}
 body{margin:0;background:var(--bg);color:var(--fg);
      font:15px/1.65 -apple-system,'Noto Sans KR',sans-serif}
 .wrap{max-width:1080px;margin:0 auto;padding:18px 20px 60px}
 .bar{display:flex;align-items:center;gap:14px;margin-bottom:14px}
 .prog{flex:1;height:6px;background:var(--card);border-radius:3px;overflow:hidden}
 .prog i{display:block;height:100%;background:var(--a);transition:width .2s}
 .meta{color:var(--dim);font-size:13px;white-space:nowrap}
 .tag{background:var(--card);border:1px solid var(--line);border-radius:4px;
      padding:2px 8px;font-size:12px;color:var(--dim)}
 .q{font-size:19px;font-weight:600;margin:16px 0 12px}
 .card{background:var(--card);border:1px solid var(--line);border-radius:8px;
       padding:16px 18px;margin-bottom:12px}
 .card h4{margin:0 0 8px;font-size:12px;color:var(--dim);font-weight:600;
          letter-spacing:.06em;text-transform:uppercase}
 .en{color:#cfd4da}
 .ko{color:var(--fg);font-size:16px}
 .frames{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;margin-bottom:12px}
 .frames img{width:100%;border-radius:5px;display:block}
 .opts{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px}
 .opt{background:var(--card);border:1px solid var(--line);border-left-width:4px;
      border-radius:7px;padding:12px 14px;cursor:pointer;transition:.12s}
 .opt:hover{background:#20242c}
 .opt b{display:block;font-size:15px;margin-bottom:3px}
 .opt small{color:var(--dim);font-size:12.5px;line-height:1.5}
 .opt kbd{float:right;background:#2a2f38;border-radius:4px;padding:1px 7px;
          font:12px monospace;color:var(--dim)}
 .o1{border-left-color:var(--b)} .o2{border-left-color:var(--a)}
 .o3{border-left-color:var(--c)} .o4{border-left-color:var(--d)}
 .after{margin-top:16px;padding:14px 16px;border-radius:8px;
        background:#16202e;border:1px solid #24425f}
 .agree{color:var(--d);font-weight:600} .disagree{color:var(--b);font-weight:600}
 .help{color:var(--dim);font-size:12.5px;margin-top:18px;line-height:1.9}
 .hide{display:none}
 .hint{margin-top:14px;padding:12px 16px;border-radius:8px;background:#1c1a16;
       border:1px solid #3a3222;color:#d8cdb4;font-size:14px}
 .hint b{color:#f0c674}
</style>
<div class=wrap>
 <div class=bar>
   <span class=tag id=ds></span>
   <div class=prog><i id=pi style="width:0"></i></div>
   <span class=meta id=cnt></span>
 </div>

 <div class=q id=q>이 설명에서 <b>정적 장면 조건</b>이 원인인가요, 배경일 뿐인가요?</div>

 <div class=frames id=fr></div>
 <div class=card><h4>정답 설명 (한국어)</h4><div class=ko id=ko></div></div>
 <div class=card><h4>원문 (English)</h4><div class=en id=en></div></div>

 <div class=opts id=opts>
   <div class="opt o1" onclick="pick(0)"><kbd>1</kbd><b>배경일 뿐 (incidental)</b>
     <small>장면이 나오지만 원인은 행위·객체에 있다. <u>장소를 바꿔도 설명이 그대로</u>면 이쪽.</small></div>
   <div class="opt o2" onclick="pick(1)"><kbd>2</kbd><b>원인의 일부 (causal)</b>
     <small>노면·기상·조명·장소가 사고 원인이나 위험도를 만든다. <u>그 조건을 빼면 설명이 달라지면</u> 이쪽.</small></div>
   <div class="opt o3" onclick="pick(2)"><kbd>3</kbd><b>장면 언급 없음 (no_scene)</b>
     <small>정적 장면 조건이 설명에 아예 등장하지 않는다.</small></div>
   <div class="opt o4" onclick="pick(3)"><kbd>4</kbd><b>이상 없음 (normal)</b>
     <small>이상·위험 사건 자체가 서술되지 않는다.</small></div>
 </div>

 <div class="after hide" id=after></div>
 <div class="hint hide" id=hint></div>

 <div class=help>
   <b>1~4</b> 판정 &nbsp;·&nbsp; <b>Enter</b> 다음 &nbsp;·&nbsp; <b>←</b> 이전 &nbsp;·&nbsp;
   <b>f</b> 프레임 접기/펼치기<br>
   판정은 <b>설명 텍스트</b> 기준입니다. 프레임은 애매할 때만 참고하세요 —
   설명에 없는 조건을 영상에서 읽어내 판정하면 기준이 흔들립니다.
 </div>
</div>
<script>
const L=['incidental','causal','no_scene','normal'];
const KO={incidental:'배경일 뿐',causal:'원인의 일부',no_scene:'장면 없음',normal:'이상 없음'};
let D=null, answered=false, t0=0, showFrames=true;
const $=id=>document.getElementById(id);

async function load(i){
  const r = await fetch('/item?idx='+i); D = await r.json();
  answered=false; t0=Date.now(); render();
}
function render(){
  $('ds').textContent = D.dataset;
  $('cnt').textContent = `${D.idx+1} / ${D.total}  ·  완료 ${D.done}`;
  $('pi').style.width = (100*D.done/D.total)+'%';
  $('ko').textContent = D.ko;
  $('en').textContent = D.en;
  $('fr').innerHTML = showFrames && D.frames[0] && !D.frames[0].startsWith('__error__')
      ? D.frames.map(s=>`<img src="${s}">`).join('') : '';
  $('opts').classList.remove('hide');
  $('after').classList.add('hide');
  if(D.prev){ $('q').innerHTML = `이 설명에서 <b>정적 장면 조건</b>이 원인인가요? <span style="color:#9aa0a6;font-size:14px">(이전 응답: ${KO[D.prev]})</span>`; }
  else { $('q').innerHTML = '이 설명에서 <b>정적 장면 조건</b>이 원인인가요, 배경일 뿐인가요?'; }
  // audit 모드: LLM 판정을 판정 전에 보여준다. 검수가 목적이므로 의도된 노출이며,
  // 이 모드의 라벨로 계산한 κ 는 독립 일치도가 아니다(서버가 mode 를 함께 기록한다).
  if(D.mode==='audit' && D.llm){
    $('hint').innerHTML = `<b>LLM 판정: ${KO[D.llm.label]||D.llm.label}</b>` +
      (D.llm.reason? `<div style="margin-top:5px;font-size:13px;color:#b0a68f">${D.llm.reason}</div>`:'') +
      `<div style="margin-top:6px;font-size:12.5px;color:#8a8270">동의하면 같은 번호를, 다르면 맞는 번호를 누르세요.</div>`;
    $('hint').classList.remove('hide');
  } else { $('hint').classList.add('hide'); }
}
async function pick(k){
  if(answered) return;
  answered = true;
  const secs = (Date.now()-t0)/1000;
  const r = await fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({clip_id:D.clip_id, label:L[k], seconds:secs})});
  const res = await r.json();
  $('opts').classList.add('hide'); $('hint').classList.add('hide');
  // LLM 판정은 여기서 처음 노출된다 — 답한 뒤라 동조가 생기지 않는다.
  let html = `<b>기록됨: ${KO[L[k]]}</b>`;
  if(res.llm){
    const same = res.llm.label === L[k];
    html += ` &nbsp;·&nbsp; LLM: ${KO[res.llm.label]||res.llm.label} ` +
            (same? '<span class=agree>일치</span>' : '<span class=disagree>불일치</span>');
    if(res.llm.reason) html += `<div style="color:#9aa0a6;font-size:13px;margin-top:6px">LLM 근거: ${res.llm.reason}</div>`;
  }
  html += `<div style="color:#9aa0a6;font-size:13px;margin-top:8px"><b>Enter</b> 또는 아무 키나 누르면 다음</div>`;
  $('after').innerHTML = html; $('after').classList.remove('hide');
}
document.onkeydown = e=>{
  const k = e.key.toLowerCase();
  if(k==='f'){ showFrames=!showFrames; render(); return; }
  // 답한 뒤에는 Enter/Space/아무 키나 다음으로. 목록 끝이면 멈춘다.
  if(answered){ e.preventDefault(); if(D.idx+1 < D.total) load(D.idx+1); else finish(); return; }
  if(k==='arrowleft'){ load(Math.max(D.idx-1,0)); return; }
  if(k==='arrowright'||k==='enter'||k===' '||k==='s'){ e.preventDefault(); load(Math.min(D.idx+1, D.total-1)); return; }
  if(['1','2','3','4'].includes(k)) pick(parseInt(k)-1);
};
function finish(){
  $('opts').classList.add('hide'); $('hint').classList.add('hide');
  $('q').textContent = '마지막 항목입니다.';
  $('after').innerHTML = `<b>완료 ${D.done} / ${D.total}</b>
    <div style="color:#9aa0a6;font-size:13px;margin-top:6px">← 로 되돌아가 수정할 수 있습니다.</div>`;
  $('after').classList.remove('hide');
}
load(0);
</script>"""


class H(BaseHTTPRequestHandler):
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
        if self.path.startswith("/item"):
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            return self._json(_payload(int(q.get("idx", ["0"])[0])))
        b = PAGE.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        cid = req["clip_id"]
        it = next(x for x in _state["items"] if x["clip_id"] == cid)
        with _lock:
            _state["labels"][cid] = {
                "clip_id": cid,
                "video_path": it["video_path"],
                "source_dataset": it["source_dataset"],
                "label": req["label"],
                "annotator": _state["annotator"],
                "seconds_spent": round(req.get("seconds", 0), 1),
                # blind = LLM 판정을 보지 않고 독립 판정 (κ 유효)
                # audit = LLM 판정을 보고 검수 (κ 무효, 라벨 품질은 더 높음)
                "mode": _state["mode"],
            }
        _save()
        return self._json({"ok": True,
                           "llm": _state["llm"].get(cid)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default=SAMPLE)
    ap.add_argument("--ko", default=KO)
    ap.add_argument("--llm", default=LLM)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--annotator", default="human_1",
                    help="두 번째 주석자는 반드시 다른 값을 주십시오(예: human_2). "
                         "같은 값이면 기존 라벨을 덮어씁니다.")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--mode", choices=("audit", "blind"), default="audit",
                    help="audit(기본): LLM 판정을 **먼저** 보여주고 검수한다. 라벨 품질이 "
                         "높아지지만 독립 판정이 아니므로 Cohen's κ 는 유효하지 않다. "
                         "blind: 답한 뒤에만 보여준다. κ 계산이 필요하면 이쪽.")
    args = ap.parse_args()

    with open(args.sample) as f:
        _state["items"] = json.load(f)
    if os.path.exists(args.ko):
        with open(args.ko) as f:
            _state["ko"] = json.load(f)
    if os.path.exists(args.llm):
        with open(args.llm) as f:
            _state["llm"] = {r["clip_id"]: {"label": r.get("label"), "reason": r.get("reason")}
                             for r in json.load(f)}
    if os.path.exists(args.out):
        with open(args.out) as f:
            _state["labels"] = {r["clip_id"]: r for r in json.load(f)}
        print(f"기존 라벨 {len(_state['labels'])}건 이어서 진행")

    _state["out"] = args.out
    _state["mode"] = args.mode
    _state["annotator"] = args.annotator

    print(f"표본 {len(_state['items'])}건 · 번역 {len(_state['ko'])}건 · "
          f"LLM 판정 {len(_state['llm'])}건 · 주석자 {args.annotator} · 모드 {args.mode}")
    if args.mode == "audit":
        print("  ⚠ audit 모드: LLM 판정을 먼저 보여줍니다. 라벨은 '검수된 라벨'이 되며,")
        print("    독립 판정이 아니므로 이 라벨로 계산한 κ 는 일치도가 아니라 동조율입니다.")
    print(f"→ http://localhost:{args.port}")
    print(f"   기록: {args.out}")
    HTTPServer(("0.0.0.0", args.port), H).serve_forever()


if __name__ == "__main__":
    main()
