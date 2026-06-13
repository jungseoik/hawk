# WebVid 데이터 준비 (다운로드 → 추출 → 학습 연결)

> 📌 **신규 연구 CERBERUS**의 Stage 1 사전학습용 WebVid 데이터 획득·전처리 절차입니다.
> **재현 목적** 문서 — 나중에 동일하게 다시 만들 수 있도록 모든 명령·함정·용량을 기록합니다.
> 환경은 `cerberus` conda env(Blackwell 호환, [`../scripts/setup_env.sh`](../scripts/setup_env.sh))를 전제합니다.

---

## 0. 배경 (왜 이런 경로로 받는가)

- 논문 Stage 1은 **WebVid-2M**(약 250만 video-caption)을 사용한다.
- **원본 WebVid는 2024-02-23 Shutterstock cease&desist로 공식 배포가 중단**되었다. `m-bain/webvid`의 URL·캡션은 더 이상 제공되지 않으며, HF의 `TempoFunk/webvid-10M`·`HuggingFaceM4/webvid` 등은 **URL+캡션 메타데이터만** 호스팅한다(실제 mp4 없음).
- 실제 **영상 바이트가 내장된** 재업로드 미러로 **`jxie/webvid_10m`**(WebVid-**10M**)을 사용한다.
  - parquet 스키마: `video: {bytes, path}` + `text: string` → 실제 mp4 + 캡션.
  - 클립: 256×256, ~3–4초, ~445KB/클립.

---

## 1. 소스 구조 (jxie/webvid_10m = WebVid-10M)

| repo | 샤드 수 | 클립(≈) | 용량 |
|---|---|---|---|
| `jxie/webvid_10m` (메인) | 439 | 87.8만 | 405 GB |
| `jxie/webvid_10m_part_0` … `part_9` | 각 ~441 | 각 ~88만 | 각 ~395 GB |
| `jxie/webvid_10m_part_10` | 330 | 66만 | 294 GB |
| **합계 (풀 10M)** | — | ~10M | **~4.65 TB** |

- 1 샤드 = `train-XXXXX-of-00439.parquet` = **2,000 클립** (~900 MB).
- `*-bootstapir_checkpoint_v2` 접미사 repo는 영상이 아니라 point-tracking 주석 → **무시**.

### 규모 선택 (필요량 → 샤드 수)
| 목표 | 클립 | 샤드 | parquet 용량 | 받는 repo |
|---|---|---|---|---|
| 스모크 | 2K | 1 | ~0.9 GB | 메인 1샤드 |
| 소규모(현재) | 20만 | 100 | ~85 GB | 메인 `train-000*` |
| **논문급 2M** | ~250만 | ~1,250 | **~1.1 TB** | 메인(439)+part_0(441)+part_1(441) |
| 풀 10M | 10M | ~4,800 | ~4.65 TB | 전체 |

---

## 2. 다운로드

### 사전: 고속 다운로더 + HF 로그인
```bash
conda activate cerberus
pip install hf_transfer          # Rust 기반 고속 다운(권장, ~36MB/s)
hf auth whoami                   # 로그인 확인 (아니면: hf auth login)
```

### 받기 (예: 소규모 100샤드 = 20만 클립)
```bash
# train-00000 ~ train-00099 (앞 100샤드)
HF_HUB_ENABLE_HF_TRANSFER=1 hf download jxie/webvid_10m --repo-type dataset \
  --include "data/train-000*.parquet" \
  --local-dir /data/pia/webvid_10m       # 큰 로컬 디스크 권장
```

### 받기 (논문급 2M = 메인 + part_0 + part_1)
```bash
for repo in webvid_10m webvid_10m_part_0 webvid_10m_part_1; do
  HF_HUB_ENABLE_HF_TRANSFER=1 hf download "jxie/$repo" --repo-type dataset \
    --include "data/*.parquet" --local-dir "/data/pia/$repo"
done
```

### ⚠️ 함정: 다운로드가 0.1MB/s로 기어가면 (resume stuck)
멈춘 연결을 이어받기(resume)하다 throttling되는 경우가 있다. **취소하고 부분파일을 지운 뒤 새 연결로** 받으면 정상 속도로 돌아온다.
```bash
pkill -f "hf download jxie"                                   # 중단
find <local-dir>/.cache -name "*.incomplete" -delete          # 부분파일 삭제
find <local-dir>/.cache -name "*.lock" -delete
# 다시 위 hf download 실행 (HF_HUB_ENABLE_HF_TRANSFER=1)
```

### ✅ 권장: stall 자동복구 watchdog (대용량·장시간 다운로드)
위 cancel/resume를 자동화한 스크립트. 바이트가 `--stall`초 동안 안 늘면 알아서 kill→부분파일 삭제→재시작한다(이미 받은 샤드는 hf가 건너뜀). 2M처럼 수 시간 걸리는 다운로드는 이걸로 무인 실행한다.
```bash
nohup python scripts/resilient_hf_download.py \
  --repos jxie/webvid_10m jxie/webvid_10m_part_0 jxie/webvid_10m_part_1 \
  --base /data/pia --include "data/*.parquet" --stall 300 --poll 20 \
  > /data/pia/watchdog.log 2>&1 &
```
- `--stall 300`: 기존 파일 해시 검증(수 분, 바이트 무성장)으로 인한 false-stall을 피하려 300초로 둔다(실 다운로드는 계속 성장하므로 영향 없음).
- 진척 확인: `ls <dir>/data/*.parquet | wc -l`, `du -sh <dir>`, `tail watchdog.log`.

---

## 3. 추출 (parquet → 디스크 mp4 + CSV)

학습 시 랜덤 액세스를 위해 parquet의 영상 바이트를 **개별 mp4 파일 + 캡션 CSV**로 추출한다(바이트 복사, 재인코딩 없음). 이렇게 하면 **기존 `WebvidDataset`을 코드 수정 0으로** 사용한다.

```bash
python scripts/extract_webvid_parquet.py \
  --parquet_dir /data/pia/webvid_10m/data \
  --out_dir     /data/pia/webvid_extracted        # videos/<page_dir>/<id>.mp4 + annotations/<page_dir>.csv
  # [--limit N]  : 총 N개만 (스모크용)
```

출력 레이아웃(= `WebvidDataset`이 기대하는 형식):
```
<out>/videos/p00000/0.mp4, 1.mp4, ...      # page_dir = "p"+샤드번호
<out>/annotations/p00000.csv               # 컬럼: page_dir, videoid, name
```

### ⚠️ 함정: page_dir은 반드시 비숫자 문자열
순수 숫자 `"00000"`은 pandas가 정수 `0`으로 파싱 → 경로가 `videos/0/..`로 깨지고 `WebvidDataset.__getitem__`이 **존재하지 않는 파일로 무한 재샘플(CPU 무한 스핀)**한다. 그래서 추출 시 `page_dir = "p"+번호`로 접두사를 붙인다(스크립트에 반영됨).

---

## 4. 추출 — `build_webvid_split.py`

> **결정: 단일 디스크 사용.** 추출본은 ~1.18TB인데 `/data/pia`에 2.3T+ 여유가 있어 용량이 제약이 아니다. 두 디스크 분산은 순전히 용량 대비책이었으므로, 단순성·I/O 일관성을 위해 **한 디스크에 모은다**. (분산 모드는 아래에 참고용으로 남김.)

### 4a. 단일 디스크 (권장)
```bash
python scripts/build_webvid_split.py --single /data/pia/webvid_extracted
```
- 3개 repo(main/part_0/part_1)를 prefix(m/a/b)로 한 폴더에 추출 → `videos/<prefix+num>/*.mp4` + `annotations/*.csv`. **union 불필요.**
- 80% 용량 가드·재시작(이미 추출된 page_dir 건너뜀) 유지.
- config: `videos_dir=/data/pia/webvid_extracted/videos`, `anno_dir=/data/pia/webvid_extracted/annotations`.
- 추출 후 parquet(1.1TB) 삭제로 공간 회수 가능.

### 4b. 두 디스크 80/20 분산 (참고용, 용량 부족 시)

용량이 부족하면 **80/20 분산 + 심볼릭 union을 한 번에** 처리하는 모드를 쓴다. WebvidDataset은 단일 `videos_dir`(+`page_dir/id.mp4`)와 `anno_dir`(CSV concat)을 읽으므로, 물리적으로 두 디스크에 나눠도 **심볼릭 union**으로 코드 수정 없이 합쳐진다.

```bash
# 라우팅 플랜만 미리 보기 (추출 안 함)
python scripts/build_webvid_split.py --dry-run

# 실제: 매 5번째 샤드(20%) -> 작은 곳(sdb2), 나머지(80%) -> 큰 곳(/data/pia)
#       각 디스크가 --cap(기본 0.80) 넘으면 그 디스크엔 중단(하드 가드)
python scripts/build_webvid_split.py     # 기본 경로: big=/data/pia/webvid_extracted,
                                         #            small=.../data/webvid_small,
                                         #            union=/data/pia/webvid_union
```
- **page_dir 고유성**: repo별 prefix(`m`=메인, `a`=part_0, `b`=part_1) → 샤드번호 충돌 방지.
- **80% 가드**: `shutil.disk_usage`로 대상 디스크 사용률이 `--cap` 이상이면 그 디스크 쓰기를 건너뛴다(양쪽 80% 초과 방지).
- **재시작 가능**: 이미 추출된 `videos/<page_dir>` + CSV는 건너뜀 → 다운로드 진행 중 부분 실행/중단 후 재실행 안전.
- 결과: `<union>/videos/<page_dir>`(양 디스크로의 심볼릭) + `<union>/annotations/*.csv`.

### 용량 계산 (2M ≈ 264만, 추출본 ~1.18TB, 80/20)
| 디스크 | 몫 | 추출본 | 피크(+parquet) | 비율 |
|---|---|---|---|---|
| 큰곳 `/data/pia`(3.4T) | 80% | ~950GB | ~2.1TB | ~62% → parquet 삭제 후 ~28% |
| 작은곳 sdb2(878G) | 20% | ~238GB | — | ~61% |
→ 양쪽 모두 80% 미만.

---

## 5. 학습 config 연결

`configs/train_configs/stage1_pretrain.yaml`의 데이터 경로를 추출 경로로 교체:
```yaml
datasets:
  webvid:
    build_info:
      anno_dir:   /data/pia/webvid_extracted/annotations/
      videos_dir: /data/pia/webvid_extracted/videos/
```
- 스모크 검증용 축소 config: [`../configs/train_configs/stage1_smoke.yaml`](../configs/train_configs/stage1_smoke.yaml) (1샤드, 5 iters).
- 실행: `bash scripts/run_stage1.sh` 또는
  `CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --cfg-path <config>`

---

## 6. 검증 기록 (재현 시 기대치)

| 항목 | 값 |
|---|---|
| 1 샤드 추출 | 2,000 mp4, ~893 MB |
| 데이터셋 `__getitem__` | ~0.4–0.5 s/clip (옵티컬 플로우 3브랜치) |
| 스모크 학습(1 GPU, batch 1, 32프레임×3) | ~1.2 s/iter, VRAM ~62 GB, 손실 5종 정상 감소 |
| 모델 | 총 10.18B 파라미터, 학습 대상 170M |

---

## 7. 디스크 위치 (이 서버 기준)

| 경로 | 종류 | 여유 | 쓰기 |
|---|---|---|---|
| `/data/pia` (sda 3.5T) | 로컬, 큰 곳 | ~3.4 TB | ✅ (`chown`으로 확보) |
| `/home/pia/seoik/hawk` (sdb2) | 로컬, 작은 곳 | ~0.5 TB | ✅ |
| `/home/pia/data/nas_192tb` | NAS | 66 TB | ✅ (NFS, 느림) |
| `/data`(루트), `nas_200tb/si_jung` 등 | — | — | ❌ 권한 |

> 대용량(2M~10M)은 `/data/pia` 권장(로컬·빠름). NAS는 백업/오버플로우용.
