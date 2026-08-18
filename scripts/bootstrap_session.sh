#!/usr/bin/env bash
# =============================================================================
# CERBERUS — 세션 재할당 복구 스크립트
# =============================================================================
# 이 컨테이너는 /home/work/seoik (NFS) 만 영구 볼륨이고, $HOME(/home/work)은
# 세션이 사라지면 통째로 초기화된다. 데이터·모델·conda·캐시는 전부 seoik 안에
# 있어서 살아남지만, 그것들을 가리키는 "연결"(심볼릭 링크, 셸 설정)은 날아간다.
#
# 새 세션을 받으면 이거 하나만 실행하면 된다:
#     bash /home/work/seoik/bootstrap_cerberus.sh
#
# 멱등(idempotent) — 여러 번 실행해도 안전하다.
# =============================================================================
set -uo pipefail

ROOT="${CERBERUS_ROOT:-/home/work/seoik}"
REPO=$ROOT/hawk
ENV_PY=$ROOT/miniconda3/envs/cerberus/bin/python
CLAUDE_DIR="$HOME/.claude"
ok(){ echo -e "  \033[0;32mOK\033[0m   $*"; }
bad(){ echo -e "  \033[0;31mFAIL\033[0m $*"; }
log(){ echo -e "\033[0;34m==>\033[0m $*"; }

# ---------------------------------------------------------------------------
log "1) Claude Code 에이전트/스킬 연결 (~/.claude — 세션마다 초기화됨)"
# 프로젝트 경로(.claude/agents)는 세션 시작 시점에만 스캔되므로, 어느 디렉토리에서
# 세션을 열든 잡히도록 전역(~/.claude)에 건다.
mkdir -p "$CLAUDE_DIR/agents" "$CLAUDE_DIR/skills"
for f in "$REPO"/.claude/agents/*.md; do
  [ -e "$f" ] && ln -sfn "$(readlink -f "$f")" "$CLAUDE_DIR/agents/$(basename "$f")"
done
for d in "$REPO"/.claude/skills/*/; do
  [ -e "$d" ] && ln -sfn "$(readlink -f "$d")" "$CLAUDE_DIR/skills/$(basename "${d%/}")"
done
# seoik_skills(별도 레포)도 있으면 함께
[ -f "$ROOT/seoik_skills/bootstrap.sh" ] && bash "$ROOT/seoik_skills/bootstrap.sh" >/dev/null 2>&1 \
  && ok "seoik_skills bootstrap 실행" || true
ok "에이전트 $(ls "$CLAUDE_DIR/agents" 2>/dev/null | wc -l)개 / 스킬 $(ls "$CLAUDE_DIR/skills" 2>/dev/null | wc -l)개 연결"

# ---------------------------------------------------------------------------
log '1.5) 대화 기록 복원 (claude --resume 을 컨테이너 이전 후에도 쓰기 위한 것)'
# ~/.claude/projects/ 가 세션마다 초기화되면 `--resume` 목록이 비어 버린다.
# 정본은 $ROOT/claude_state 에 두고, 여기서 되돌린 다음 주기적으로 다시 저장한다.
SYNC="$REPO/scripts/claude_state_sync.sh"
if [ -x "$SYNC" ]; then
  bash "$SYNC" --restore | tail -1
  bash "$SYNC" --daemon 300 | tail -1
else
  bad "claude_state_sync.sh 없음 — 대화 기록이 컨테이너와 함께 사라진다"
fi

# ---------------------------------------------------------------------------
log "2) 셸 환경 (~/.bashrc — 세션마다 초기화됨)"
# 컨테이너가 자체 conda(/home/work/miniconda3)를 base로 활성화하고 envs_dirs를
# 자기 것으로 고정하므로, `conda run -n cerberus`는 엉뚱한 경로로 풀린다.
# 항상 절대 prefix로 접근할 것 — 그래서 편의 변수를 심어둔다.
if ! grep -q "CERBERUS bootstrap" ~/.bashrc 2>/dev/null; then
  cat >> ~/.bashrc <<'EOF'

# >>> CERBERUS bootstrap >>>
export CERBERUS_ROOT=/home/work/seoik
export CERBERUS_PY=$CERBERUS_ROOT/miniconda3/envs/cerberus/bin/python
export TORCH_HOME=$CERBERUS_ROOT/cache/torch      # eva_vit_g / blip2 qformer
export HF_HOME=$CERBERUS_ROOT/cache/hf            # bert-base-uncased, tokenizers
[ -f $CERBERUS_ROOT/.hf_token ] && export HF_TOKEN=$(cat $CERBERUS_ROOT/.hf_token)
[ -f $CERBERUS_ROOT/.gemini_token ] && export GEMINI_API_KEY=$(cat $CERBERUS_ROOT/.gemini_token)  # GPT-guided 판정자

# 생성 결정론에 필요하다. `do_sample=False` 만으로는 부족했다 — 일부 CUDA 커널이
# 비결정적이고 fp16 에서 그 차이가 argmax 를 뒤집으면 이후 토큰이 전부 갈라진다.
# 실측: 같은 체크포인트·같은 클립을 두 번 평가해 3/3 다른 문장, 하나는 `4 4 4 4 …` 로 퇴화.
# 이 변수는 CUDA 컨텍스트 생성 **전**에 설정되어야 하므로 셸 환경에 둔다.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
alias cpy='$CERBERUS_PY'
alias cdh='cd $CERBERUS_ROOT/hawk'
# 주의: 컨테이너 자체 conda가 `-n cerberus`를 가로챈다. 반드시 -p 절대경로로:
#   $CERBERUS_ROOT/miniconda3/bin/conda run -p $CERBERUS_ROOT/miniconda3/envs/cerberus ...
# <<< CERBERUS bootstrap <<<
EOF
  ok "~/.bashrc에 CERBERUS 블록 추가"
else
  ok "~/.bashrc 이미 설정됨"
fi

if ! git config --global user.email >/dev/null 2>&1; then
  git config --global user.name "jungseoik"; git config --global user.email "si.jung@pia.space"
  ok "git 사용자 설정"
else ok "git 사용자 이미 설정됨 ($(git config --global user.email))"; fi

# ---------------------------------------------------------------------------
log "3) 영구 자산 검증 (seoik 볼륨 — 살아있어야 정상)"
chk_dir(){ [ -d "$2" ] && ok "$1: $2" || bad "$1 없음: $2"; }
chk_dir "레포"           "$REPO"
chk_dir "conda env"      "$ROOT/miniconda3/envs/cerberus"
chk_dir "캐시(torch/hf)" "$ROOT/cache"
chk_dir "LLaMA 가중치"   "$REPO/weights/Video-LLaMA-2-7B-Finetuned/llama-2-7b-chat-hf"
chk_dir "WebVid 추출본"  "$ROOT/webvid_extracted/videos"
chk_dir "이상행동 데이터" "$ROOT/hawk_anomaly/Videos"
chk_dir "학습 run"       "$ROOT/runs/core/main"
[ -f "$ROOT/.hf_token" ] && ok "HF 토큰" || bad "HF 토큰 없음"
[ -f "$ROOT/.github_token" ] && ok "GitHub 토큰" || bad "GitHub 토큰 없음"
[ -f "$ROOT/.gemini_token" ] && ok "Gemini 토큰 (판정자)" || echo "  --   Gemini 토큰 없음 (GPT-guided 평가 불가, 나머지는 정상)"

LAST=$(ls "$ROOT"/runs/core/main/checkpoint_*.pth 2>/dev/null | sort -V | tail -1)
[ -n "$LAST" ] && ok "최신 체크포인트: $(basename "$LAST") (총 $(ls "$ROOT"/runs/core/main/checkpoint_*.pth 2>/dev/null | wc -l)개)" \
               || bad "체크포인트 없음"
SH=$(ls "$ROOT/webvid_extracted/annotations"/*.csv 2>/dev/null | wc -l)
ok "WebVid shard $SH/1330,  이상행동 비디오 $(find "$ROOT/hawk_anomaly/Videos" -type f 2>/dev/null | wc -l)개"

# ---------------------------------------------------------------------------
log "4) 런타임 확인"
if [ -x "$ENV_PY" ]; then
  "$ENV_PY" - <<'PY' 2>/dev/null || bad "torch import 실패"
import torch, transformers
n = torch.cuda.device_count() if torch.cuda.is_available() else 0
print(f"  \033[0;32mOK\033[0m   torch {torch.__version__} / transformers {transformers.__version__} / GPU {n}장")
PY
else bad "cerberus env python 없음: $ENV_PY"; fi

echo
log "완료. 새 셸에서 적용하려면:  source ~/.bashrc"
echo "   현황·다음 단계:  $ROOT/README.md   ·   $REPO/docs/training-log.md"
echo "   이전 대화 이어가기:  claude --continue   (또는 claude --resume 으로 목록 선택)"
