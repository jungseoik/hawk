# 근거 색인 — 어떤 주장이 무엇으로 뒷받침되는가

논문과 문서의 각 주장에 대해 **어디를 보면 근거를 확인할 수 있는지**를 모은다.
"그건 왜 그렇게 정했나"라는 질문이 나올 때 이 표에서 출발하면 된다.

재현 명령이 있는 항목은 그대로 실행해 확인할 수 있다. `$CERBERUS_PY`는 부트스트랩
(`bash $CERBERUS_ROOT/bootstrap_cerberus.sh`) 후 사용 가능하다.

---

## 1. 파이프라인 결함과 수정 (전부 실측으로 확인)

| 주장 | 근거 | 재현 |
|---|---|---|
| 세 스트림에 **독립적인 crop**이 걸려 상보성 항등식이 깨졌다 | 크롭 파라미터 3회 호출이 각각 다른 값: `(0,80,360,480)` / `(31,120,320,394)` / `(5,71,343,350)`. 항등식 위반 화소 **77.6–88.5%** | `tests/test_stream_alignment.py::test_augmentation_uses_one_shared_crop` |
| 수정 후 위반 화소가 **1.4–3.3%** 로 떨어졌다 (잔차는 리샘플링 기인) | 12 seed 측정. 두 분포가 3.3%와 77.6% 사이에서 완전히 갈림 | 같은 테스트 (임계값 8%가 두 원인을 가르는 경계) |
| Stage-1은 **프레임 인덱스까지 달랐다** | `load_video`와 `load_video_motion_and_background`를 `sampling="headtail"`(난수)로 각각 호출 | `git show 82f2854 -- hawk/processors/video_processor.py` |
| `frame_list[0] = 1`이 **첫 프레임만** 다른 시점으로 만들었다 | frame 0 최대 오차 18, 나머지 프레임 0.000 | 수정 후 디코딩 단계 `m + b == a` 완전 일치 |
| `ToUint8`가 클램프가 아니라 **wraparound**한다 | `[-3.7, -1.2, 258.9, 300.0] → [253, 255, 2, 44]`. 마스크된 스트림에서만 범위 이탈(motion 0.52% / background 0.36%, appearance 0.000%) | `$CERBERUS_PY -c "import torch; print(torch.tensor([-3.7,300.0]).to(torch.uint8))"` |
| **추론 경로가 dual-branch로 실행되고 있었다** | `upload_video_without_audio`가 외형·동적만 concat. 시각 토큰 64개(학습은 96개) | `git show 85e5d1a -- hawk/conversation/conversation_video.py` |
| 생성이 **비결정론적**이었다 | `do_sample=True` 하드코딩. 같은 클립 2회 생성 시 출력 불일치 | 수정 후 2회 실행 결과 완전 일치 확인 |
| CUDA RNG 접근이 **DataLoader 워커를 죽인다** | fork 워커에서 `is_available()=True`, `is_initialized()=False` → `Cannot re-initialize CUDA in forked subprocess` | `tests/test_stream_alignment.py::test_transform_works_inside_forked_worker` |

---

## 2. `L_dis` 퇴화 (기여 제외의 근거)

| 주장 | 측정값 | 재현 |
|---|---|---|
| 손실이 **입력 무관 편향만으로** 만족되었다 | `encoder_1.bias` 코사인 `+0.9988` / `−0.9993` | 아래 스니펫 |
| 대조군은 무작위 수준이다 | 같은 모듈 `decoder_2.bias` `+0.027` / `+0.023` | 〃 |
| **가중치는 학습되지 않았다** | `encoder_1.weight` 노름이 초기화 대비 0.97 / 1.00 / 1.01배, 편향은 19.9 / 27.8 / 40.6배 | 〃 |
| 이후 **기울기가 0**이다 | `cos = ±1`은 코사인의 정류점. 로그의 `middleloss`가 첫 epoch 이후 `0.0000` 유지 | `grep "Averaged stats" $CERBERUS_ROOT/runs/core/train.log` |

```bash
$CERBERUS_PY - <<'PY'
import torch, torch.nn.functional as F
sd = torch.load('/home/work/seoik/runs/core/main/checkpoint_106.pth', map_location='cpu')['model']
A,M,B = 'llama_proj_last','llama_proj_motion_last','llama_proj_background_last'
cos = lambda x,y: F.cosine_similarity(x.float().flatten(), y.float().flatten(), 0).item()
b = [sd[f'{p}.encoder_1.bias'] for p in (A,M,B)]
d = [sd[f'{p}.decoder_2.bias'] for p in (A,M,B)]
print('encoder_1.bias  cos(a,m)=%+.4f cos(m,b)=%+.4f' % (cos(b[0],b[1]), cos(b[1],b[2])))
print('decoder_2.bias  cos(a,m)=%+.4f cos(m,b)=%+.4f  (대조군)' % (cos(d[0],d[1]), cos(d[1],d[2])))
for n,p in [('appearance',A),('motion',M),('background',B)]:
    W, bb = sd[f'{p}.encoder_1.weight'], sd[f'{p}.encoder_1.bias']
    print(f'  {n:11s} |W|={W.norm():8.3f}  |b|={bb.norm():7.4f}')
PY
```

---

## 3. 지표의 결함과 보정

| 주장 | 근거 | 재현 |
|---|---|---|
| **CDS가 `L_dis` 최적점에서 0이 된다** | linear CKA는 부호 불변 → `z_b = −c·z_m`에서 Redundancy = 1. 측정: `cos=−1`일 때 CDS 0.0000, `cos≈0`일 때 0.5409 | `experiments/disentanglement.py` |
| **Coverage가 지배 블록 하나만 잰다** | 정적 블록 노름 40배 → Coverage 0.9622 = `CKA(z_b, ref)` 단독값과 **정확히 일치**. 블록 정규화 후 0.8189 | 〃 (`block_normalize` 인자) |
| **BSI가 두 해석을 구분하지 못한다** | 배경 의미 사용: signal 0.750 / null 0.286 → corrected **+0.464**. 픽셀 민감: 0.917 / 0.917 → **0.000**. 보정 전에는 후자가 더 높음 | `experiments/bg_critical_benchmark/background_swap.py::corrected_bsi` |
| **붕괴가 손실값으로는 안 보인다** | 스트림 내부 rank는 정상(123.6)이므로 rank만으로는 탐지 불가. 공선성 비율이 주 지표 | `disentanglement.py::collapse_diagnostics` |
| **Scene-word Recall이 주장을 직접 잰다** | 배경 서술 응답 0.923 vs 움직임만 서술 0.077 (동일 정답, 장면 어휘 13개) | `scripts/evaluate.py::scene_word_recall` |

---

## 4. 데이터 통계

| 주장 | 근거 |
|---|---|
| 마스크 비율이 도메인 간 **10배 이상** 차이 난다 | 전수 7,852 클립 측정: DoTA 0.449 / Ped2 0.160 / Ped1 0.159 / Avenue 0.147 / ShanghaiTech 0.067 / UCF-Crime 0.064 / UBnormal 0.039 → `experiments/bg_critical_benchmark/mask_statistics.json` |
| DoTA가 전체의 **62%**이며 자기운동으로 마스크가 퇴화한다 | 4,883 / 7,852 클립 |
| 측정은 **학습 파이프라인과 동일 방식**이다 | 인접 프레임(i−1, i) 플로우. 샘플 프레임 간 플로우로 재면 과대 추정(UCF-Crime 0.072 → 0.433) |
| annotation 경로가 원저자 서버 절대경로였다 | `os.path.join(vis_root, abs_path)`는 `vis_root`를 버린다 → `scripts/localize_anomaly_annotations.py` (train 7066 / test 786 / all 7852, 누락 0) |

재현: `$CERBERUS_PY scripts/curate_bg_critical.py --workers 24`

---

## 5. 서버 환경 함정

| 함정 | 증상 | 근거 |
|---|---|---|
| `conda run -n cerberus`가 엉뚱한 경로로 풀린다 | `No module named 'torch'` | 루트 `CLAUDE.md` §3.1 |
| 환경에 `NPROC=42`가 미리 설정돼 있다 | torchrun이 GPU 42장 요구 → `invalid device ordinal` | 〃 §3.5, `git show 5ce1834` |
| NFS 소파일 쓰기가 ~50 files/s에서 포화 | 워커를 늘려도 안 빨라짐 | 〃 §3.2 |
| 데이터셋의 bare `except:`가 진짜 예외를 삼킨다 | `NameError`가 "비디오 로드 실패"로 보고됨 | `git show 1331150` |

---

## 5.5 부록으로 내린 것과 그 부활 조건

논문에서 어떤 내용이 **왜** 부록으로 갔는지, 그리고 무엇이 그것을 다시 본문으로 올리는지는
나중에 재구성하기 어렵다. 의존관계를 여기 남긴다.

| 부록 | 내용 | 원래 목적 | 부활 조건 |
|---|---|---|---|
| A | `L_dis` 퇴화 사후 분석 전문 | 목적함수 진단 | — (기록으로 영구 보존) |
| C | CDS 정의·측정 주의 | `L_dis` 달성도 측정 | 목적함수 수정 + 부호/기준축 문제 해결 |
| D | E1 표현 수준 진단 | CDS로 분리 측정 | ③에서 분리도가 실제로 변할 때 |
| E | E3 분리 목적함수 검정 | 손실 방향의 인과 검증 | 동상 (축은 이미 "퇴화하지 않는 목적함수 vs 없음"으로 교체됨) |
| B·F·G | 효율·손실 절제·τ 스윕 | 보조 실험 | 지면이 허용할 때 |

상세와 되살릴 때 함께 고쳐야 할 것은 `experiment-roadmap.md` §2의 "③이 되살리는 것" 참조.
**어느 경우에도 삭제하지 말 것** — 음성 결과로 확정되어도 후속 연구가 같은 설계를 반복하지
않게 하는 기록이다.

---

## 6. 의사결정 기록

| 결정 | 근거 | 문서 |
|---|---|---|
| Stage-2 `batch 2` / `max_epoch 160` | 150 iter 공정 비교에서 처리량이 batch에 무관(2.92 / 2.89 / 2.78 samples/s), batch 8은 OOM. 원본과 샘플 예산·effective batch 동시 일치 | `training-log.md` |
| 어블레이션 감축 예산 40 epoch | 통제 논리는 "arm끼리 같은 예산"이면 성립 | `experiment-roadmap.md` §1 |
| `L_dis`·CDS를 기여에서 제외 | 위 §2 | `improved/01_introduction.md`, `03_methodology.md` §3.1.3 |
| Stage-1 재학습을 후순위로 | `encoder_1`이 곁가지라 Stage-2만으로 검증 가능(1.6일 vs 9.1일) | `experiment-roadmap.md` §2 |
| 판정자를 Gemini로 | BLEU가 논문 대 논문 비교를 담당하고, 판정 지표는 동일 판정자 내 비교로 한정 | `scripts/evaluate.py::judge_gpt_guided` |

---

## 7. 심사 대응

2석 패널(Journal-Fit + Methodology) 심사 결과와 항목별 대응, 그리고 코드로 재확인한
사실들은 `review-2026-08-10-methodology.md`에 있다. 재검증 라운드에서 추가로 드러난
표현 붕괴와 CDS 모순도 같은 문서에 기록되어 있다.
