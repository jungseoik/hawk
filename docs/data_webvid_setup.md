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

## 4. 두 디스크 분산 (큰 곳 + 작은 곳) — 선택

WebvidDataset은 단일 `videos_dir`(+`page_dir/id.mp4`)와 `anno_dir`(CSV 전부 concat)을 읽는다. 물리적으로 두 디스크에 나눠도 **심볼릭링크 union**으로 코드 수정 없이 합칠 수 있다.
```bash
# 예: 일부 샤드는 /data/pia(큰 곳), 일부는 sdb2(작은 곳)에 추출한 뒤
mkdir -p /data/pia/webvid_union/videos /data/pia/webvid_union/annotations
ln -s /data/pia/webvid_extracted/videos/p000{00..79}        /data/pia/webvid_union/videos/    # 큰 곳
ln -s /home/pia/seoik/hawk/data/webvid_small/videos/p000{80..99} /data/pia/webvid_union/videos/  # 작은 곳
cp /data/pia/webvid_extracted/annotations/*.csv  /data/pia/webvid_union/annotations/
cp /home/pia/seoik/hawk/data/webvid_small/annotations/*.csv /data/pia/webvid_union/annotations/
# config의 videos_dir=/data/pia/webvid_union/videos, anno_dir=/data/pia/webvid_union/annotations
```

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
