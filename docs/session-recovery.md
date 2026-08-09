# CERBERUS 작업 볼륨 — 새 세션이라면 여기부터

이 컨테이너는 **`/home/work/seoik` (NFS 9.1T) 만 영구**입니다.
`$HOME`(`/home/work`)은 세션이 사라지면 통째로 초기화됩니다.

## 새 세션에서 첫 명령

```bash
bash /home/work/seoik/bootstrap_cerberus.sh   # 연결 복구 + 자산 검증 (멱등)
source ~/.bashrc
```

이거 하나로 Claude 에이전트/스킬 심링크, 셸 환경(`CERBERUS_PY`·`TORCH_HOME`·`HF_HOME`·`HF_TOKEN`),
git 설정이 복구되고, 데이터·모델·체크포인트가 살아있는지까지 검증합니다.

---

## 지금까지 진행 상황 (2026-08-10 기준)

| 단계 | 상태 |
|---|---|
| WebVid 다운로드 + mp4 추출 | ✅ 1330/1330 shard, **263.8만 clip** |
| 환경·가중치·모델 빌드 검증 | ✅ torch 2.11+cu128 / transformers 4.28 / H100 3장 |
| **Stage-1 사전학습** | ✅ **완주** — epoch 106, LR→min_lr(1e-6), loss 3.815→2.8506 |
| Stage-2 이상행동 데이터 | ✅ 7,899개 / 121GB (HAWK 7종) |
| 논문 도구 | ✅ 자체 에이전트 4 + 외부 스킬 4(vendor) |
| **E1/E2 실측** | ⏳ **미실행** (합성 데이터 자체검증만 완료) |
| **Stage-2 finetune** | ⏳ **미시작** (데이터·config 준비 완료) |
| 논문 원고 | ◐ 5개 장 작성, 결과·conclusion·references 미작성 |

**Stage-1 결과 요약**: `cos(z_a,z_m)`·`cos(z_m,z_b)` 모두 **0.0000 수렴** — CVD 상보성(C1/C2)의 직접 증거.
상세: [`runs/core/STOPPED.md`](runs/core/STOPPED.md), [`hawk/docs/training-log.md`](hawk/docs/training-log.md),
곡선: `hawk/figs/stage1_curves.{png,pdf,csv}`

---

## 자산 지도 (전부 이 볼륨 안, 3.7T 사용 / 5.5T 여유)

```
/home/work/seoik/
├── hawk/                     git 레포 (github.com/jungseoik/hawk)
│   ├── weights/              LLaMA-2-7B-chat 13GB
│   ├── figs/stage1_curves.*  논문용 학습 곡선
│   ├── vendor/               외부 논문 스킬 (CC BY-NC 4.0)
│   └── .claude/{agents,skills,agent-memory}
├── miniconda3/envs/cerberus/ 학습 환경 (python 3.10, torch cu128)
├── cache/{torch,hf}/         eva_vit_g · blip2 qformer · bert
├── runs/core/main/           checkpoint_0 ~ 106 (54개) + TensorBoard 전 구간
├── webvid_extracted/         Stage-1 학습 데이터 263.8만 mp4
├── webvid_10m{,_part_0,_part_1}/  원본 parquet 1.1TB (추출 완료 — 삭제 가능)
├── hawk_anomaly/             Stage-2 이상행동 7종 121GB
├── seoik_skills/             별도 스킬 레포
├── .hf_token / .github_token
└── bootstrap_cerberus.sh     ← 세션 복구
```

> `webvid_10m*` 1.1TB는 추출이 끝나 **삭제해도 됩니다**. 재다운로드는
> `hawk/scripts/resilient_hf_download.py`로 가능(약 7시간).

---

## 이 서버 특유의 함정 (재발 방지)

1. **`conda run -n cerberus` 는 실패합니다.** 컨테이너가 자체 conda(`/home/work/miniconda3`)를
   base로 활성화하고 `envs_dirs`를 자기 것으로 고정해서, 이름으로 찾으면 엉뚱한 경로로 풀립니다.
   → **항상 절대 prefix**: `conda run -p /home/work/seoik/miniconda3/envs/cerberus ...`
   또는 `$CERBERUS_PY` 직접 실행. (`train_run.sh`·`setup_env.sh`는 이미 수정됨)
2. **`$HOME`에 아무것도 두지 마세요.** 캐시(`TORCH_HOME`/`HF_HOME`)도 seoik로 고정돼 있습니다.
3. **NFS가 유일한 병목**입니다. CPU 83코어·RAM 676GB는 남아돕니다. 소파일 쓰기가
   **병렬도와 무관하게 ~50 files/s에서 포화**하므로 워커를 늘려도 빨라지지 않습니다.
4. **Claude 에이전트는 세션 시작 시점에만 스캔**됩니다. 중간에 추가하면 한동안 안 잡히니
   부트스트랩을 먼저 돌리세요.

---

## 다음 작업 (둘 다 GPU 필요, 현재 3장 유휴)

**① E1/E2 실측** (짧음)
```bash
cd /home/work/seoik/hawk
$CERBERUS_PY scripts/extract_representations.py --cfg configs/eval_configs/eval.yaml \
    --ckpt /home/work/seoik/runs/core/main/checkpoint_106.pth --out experiments/out/reps.npz
$CERBERUS_PY experiments/disentanglement.py --reps experiments/out/reps.npz   # CDS
```

**② Stage-2 finetune** (장시간)
`configs/train_configs/stage2_finetune.yaml` 에서
- `ckpt:` → `/home/work/seoik/runs/core/main/checkpoint_106.pth`
- `anno_dir:` → `/home/work/seoik/hawk_anomaly/Annotation/All_Mix/all_videos_train.json`
- `videos_dir:` → `/home/work/seoik/hawk_anomaly/Videos/`
```bash
bash scripts/train_run.sh configs/train_configs/stage2_finetune.yaml stage2 0,1,2 3
```

---

## 백업 상태 (다른 서버에서도 복원 가능)

- **GitHub** `jungseoik/hawk` — 코드·문서·figure·vendor 스킬 전부 push 완료
- **HF** `backseollgi/Cerberus/stage1_core/` — checkpoint 0~106(마일스톤+최종),
  `train.log`, `config.yaml`, `STOPPED.md`, `training-log.md`, `stage1_curves.png/csv`
- 새 서버 재현 절차: [`hawk/docs/reproduce.md`](hawk/docs/reproduce.md) ·
  서버 이전: [`hawk/docs/MIGRATION.md`](hawk/docs/MIGRATION.md)
