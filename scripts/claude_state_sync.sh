#!/usr/bin/env bash
# =============================================================================
# Claude Code 세션 상태 영속화 — `--resume` 를 컨테이너 이전 후에도 쓰기 위한 것
# =============================================================================
# `claude --resume` / `--continue` 는 ~/.claude/projects/<cwd-slug>/*.jsonl 을
# 읽어 대화를 복원한다. 그런데 이 컨테이너에서 $HOME 은 세션마다 초기화되므로
# 그 디렉토리가 통째로 사라진다 → 재개 목록이 비어 버린다.
#
# 그래서 영구 볼륨(seoik)에 정본을 두고 양방향으로 옮긴다.
#
#   --restore        영구 저장소 → ~/.claude   (새 컨테이너 부팅 직후 1회)
#   --save           ~/.claude → 영구 저장소   (지금 상태를 박아둔다)
#   --daemon [초]    --save 를 주기적으로 반복 (기본 300초). 컨테이너가 예고 없이
#                    죽어도 최대 그 간격만큼만 잃는다
#   --stop-daemon    데몬 정지
#   --status         양쪽 파일 수·최신 시각·데몬 상태
#
# 심링크를 쓰지 않는 이유: 세션이 열려 있는 동안 파일 핸들이 살아 있어서
# 디렉토리를 파일시스템 경계 넘어 옮기면(overlay → NFS) 진행 중인 대화 기록이
# 삭제된 inode 로 흘러 들어간다. 복사 방식이 그 위험이 없다.
# =============================================================================
set -uo pipefail

ROOT="${CERBERUS_ROOT:-/home/work/seoik}"
STORE="$ROOT/claude_state"          # 영구 정본
LIVE="$HOME/.claude"                # 휘발성 실사용 위치
PROJ="projects/-home-work-seoik"    # cwd=/home/work/seoik 세션의 기록
PIDF="$STORE/.sync.pid"
LOGF="$STORE/.sync.log"
INTERVAL_DEFAULT=300

ok(){ echo -e "  \033[0;32mOK\033[0m   $*"; }
log(){ echo -e "\033[0;34m==>\033[0m $*"; }

# cp -au: 대상이 더 새로우면 건드리지 않는다(양방향 안전). 삭제는 절대 하지 않는다.
copy_tree(){ # copy_tree <src-dir> <dst-parent>
  [ -d "$1" ] || return 0
  mkdir -p "$2"
  cp -au "$1" "$2"/ 2>/dev/null
}
copy_file(){ # copy_file <src-file> <dst-file>
  [ -f "$1" ] || return 0
  mkdir -p "$(dirname "$2")"
  cp -au "$1" "$2" 2>/dev/null
}

count(){ find "$1" -name '*.jsonl' 2>/dev/null | wc -l; }
newest(){ find "$1" -name '*.jsonl' -printf '%TY-%Tm-%Td %TH:%TM  %f\n' 2>/dev/null | sort -r | head -1; }

do_save(){
  mkdir -p "$STORE/$PROJ"
  copy_tree "$LIVE/$PROJ"          "$STORE/projects"
  copy_file "$LIVE/history.jsonl"  "$STORE/history.jsonl"
  copy_tree "$LIVE/todos"          "$STORE"
  copy_tree "$LIVE/shell-snapshots" "$STORE"
  copy_file "$LIVE/settings.json"  "$STORE/settings.json"
}

do_restore(){
  mkdir -p "$LIVE/projects"
  copy_tree "$STORE/$PROJ"          "$LIVE/projects"
  copy_file "$STORE/history.jsonl"  "$LIVE/history.jsonl"
  copy_tree "$STORE/todos"          "$LIVE"
  copy_tree "$STORE/shell-snapshots" "$LIVE"
}

daemon_running(){ [ -f "$PIDF" ] && kill -0 "$(cat "$PIDF" 2>/dev/null)" 2>/dev/null; }

case "${1:---status}" in
  --restore)
    log "영구 저장소 → ~/.claude 복원"
    do_restore
    ok "세션 기록 $(count "$LIVE/$PROJ")개 사용 가능  ($LIVE/$PROJ)"
    ;;
  --save)
    do_save
    ok "저장: 세션 $(count "$STORE/$PROJ")개  ($STORE/$PROJ)"
    ;;
  --daemon)
    if daemon_running; then ok "데몬 이미 실행 중 (pid $(cat "$PIDF"))"; exit 0; fi
    IV="${2:-$INTERVAL_DEFAULT}"
    mkdir -p "$STORE"
    setsid bash -c '
      STORE="'"$STORE"'"; IV="'"$IV"'"; SELF="'"$(readlink -f "$0")"'"
      echo $$ > "$STORE/.sync.pid"
      trap "rm -f $STORE/.sync.pid; exit 0" TERM INT
      while :; do
        bash "$SELF" --save >>"$STORE/.sync.log" 2>&1
        sleep "$IV"
      done' </dev/null >/dev/null 2>&1 &
    sleep 1
    daemon_running && ok "데몬 시작 (pid $(cat "$PIDF"), ${IV}초 주기)" || echo "  데몬 시작 실패"
    ;;
  --stop-daemon)
    if daemon_running; then kill "$(cat "$PIDF")"; rm -f "$PIDF"; ok "데몬 정지"; else ok "데몬 미실행"; fi
    ;;
  --status)
    echo "영구 저장소  $STORE/$PROJ"
    echo "  세션 $(count "$STORE/$PROJ")개 · 최신 $(newest "$STORE/$PROJ")"
    echo "실사용      $LIVE/$PROJ"
    echo "  세션 $(count "$LIVE/$PROJ")개 · 최신 $(newest "$LIVE/$PROJ")"
    daemon_running && echo "데몬        실행 중 (pid $(cat "$PIDF"))" || echo "데몬        미실행"
    ;;
  *) echo "사용법: $0 [--restore|--save|--daemon [초]|--stop-daemon|--status]"; exit 2 ;;
esac
