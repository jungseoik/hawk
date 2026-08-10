"""세 스트림(외형 / 동적 / 정적) 정합에 대한 회귀 테스트.

논문의 최상위 기여 C1은 상보성 항등식 `M ⊙ x + (1 − M) ⊙ x = x` 이다. 이 등식을
코드가 실제로 만족하는지 확인하는 테스트가 없었기 때문에, 아래 세 가지가 모두
발견되지 않은 채 Stage-1 전체(107 epoch)를 통과했다.

  1. 세 스트림에 독립적인 RandomResizedCrop이 적용되어 서로 다른 영역이 잘렸다.
  2. Stage-1은 `sampling="headtail"` 난수 추출을 두 번 호출해 외형 스트림과
     동적/정적 스트림의 프레임 인덱스 자체가 달랐다.
  3. `compute_motion_and_background`가 `frame_list[0] = 1`로 첫 인덱스를 덮어써,
     동적/정적 스트림의 첫 프레임만 외형과 다른 시점이었다.

실행:
    $CERBERUS_PY -m pytest tests/test_stream_alignment.py -v
"""
import glob
import os

import pytest
import torch

from hawk.processors.video_processor import (
    AlproVideoEvalProcessor,
    AlproVideoTrainProcessor,
    load_streams_aligned,
    load_video,
    load_video_motion_and_background,
)

ROOT = os.environ.get("CERBERUS_ROOT", "/home/work/seoik")
VIDEO_GLOBS = [
    f"{ROOT}/hawk_anomaly/Videos/ShanghaiTech/testing/videos/*.mp4",
    f"{ROOT}/hawk_anomaly/Videos/DoTA/Video/*.mp4",
    f"{ROOT}/hawk_anomaly/Videos/UCF_Crime/videos/*.mp4",
]


def _sample_videos():
    paths = []
    for pattern in VIDEO_GLOBS:
        found = sorted(glob.glob(pattern))
        if found:
            paths.append(found[0])
    if not paths:
        pytest.skip("이상행동 비디오를 찾을 수 없습니다 (hawk_anomaly 미존재)")
    return paths


@pytest.mark.parametrize("vpath", _sample_videos())
def test_decomposition_is_exact(vpath):
    """디코딩 직후 motion + background 가 원본과 화소 단위로 정확히 일치해야 한다."""
    appearance = load_video(video_path=vpath, n_frms=8, height=224, width=224).float()
    motion, background = load_video_motion_and_background(
        video_path=vpath, n_frms=8, height=224, width=224
    )
    assert torch.equal(motion.float() + background.float(), appearance), (
        "상보성 항등식 위반: motion + background != appearance. "
        "프레임 인덱스 정합 또는 마스크 구성이 깨졌습니다."
    )


@pytest.mark.parametrize("vpath", _sample_videos()[:1])
def test_streams_share_frame_indices_under_random_sampling(vpath):
    """난수 기반 headtail 샘플링에서도 세 스트림의 프레임이 일치해야 한다."""
    appearance, motion, background = load_streams_aligned(
        vpath, n_frms=8, image_size=224, sampling="headtail"
    )
    assert torch.equal(motion.float() + background.float(), appearance.float())


@pytest.mark.parametrize(
    "processor_cls", [AlproVideoTrainProcessor, AlproVideoEvalProcessor]
)
def test_augmentation_uses_one_shared_crop(processor_cls):
    """증강 이후에도 세 스트림이 같은 영역을 봐야 한다.

    RandomResizedCrop 은 호출마다 새 파라미터를 뽑으므로, 스트림마다 transform 을
    따로 호출하면 서로 다른 영역이 잘린다. 이 테스트는 잔차의 **크기 등급**으로 두
    원인을 구분한다.

      - 크롭 불일치: 스트림이 서로 다른 영역을 보므로 오차가 이미지 전역에 퍼진다.
        실측(수정 전 방식, 8 seeds) **77.6 ~ 88.5%**. 이것이 막으려는 회귀다.
      - 리샘플링 잔차: 크롭이 정합된 상태에서도 bicubic 보간이 마스크 경계에서
        값을 범위 밖으로 밀고, 그것을 스트림별로 각각 클램프하기 때문에
        `clamp(m) + clamp(b) != clamp(a)` 가 된다. 경계에 국한되며 실측(12 seeds)
        **1.39 ~ 3.34%** (평균 오차 자체는 0.11%). 완전 제거하려면 마스킹을 crop
        이후로 옮겨야 한다 (미적용 — docs/review-2026-08-10-methodology.md 참조).

    두 분포가 3.3% 와 77.6% 사이에서 완전히 갈리므로, 임계값 8% 는 "정확함"의
    기준이 아니라 **두 원인을 가르는 경계**다. 크롭 강도에 따라 잔차가 달라지므로
    시드를 고정해 결정론적으로 만든다.
    """
    vpath = _sample_videos()[0]
    processor = processor_cls(image_size=224, n_frms=8)
    torch.manual_seed(0)
    appearance, motion, background = processor(vpath)

    residual = (motion.float() + background.float() - appearance.float()).abs()
    value_range = float(appearance.max() - appearance.min())
    off_pixels = (residual > 0.01 * value_range).float().mean().item()

    assert off_pixels < 0.08, (
        f"{processor_cls.__name__}: 화소 {off_pixels:.1%} 가 어긋났습니다 — "
        "리샘플링 잔차 수준을 넘었으므로 세 스트림이 서로 다른 crop 을 받고 있을 "
        "가능성이 높습니다."
    )


@pytest.mark.parametrize(
    "ablation,expected_static_coverage",
    [("flow", None), ("random_mask", None), ("duplicate", 1.0), ("zero", 0.0)],
)
def test_static_stream_ablations(ablation, expected_static_coverage):
    """통제군 네 조건이 의도한 정적 스트림 내용을 만들어야 한다.

    핵심은 random_mask 가 flow 와 **같은 면적비**를 유지하는 것이다. 면적이 다르면
    "내용"이 아니라 "정보량"을 비교하게 되어 통제가 성립하지 않는다.
    """
    vpath = _sample_videos()[0]
    motion, background = load_video_motion_and_background(
        video_path=vpath, n_frms=8, height=224, width=224, ablation=ablation
    )
    coverage = (background.float().sum(0) > 0).float().mean().item()

    if expected_static_coverage is not None:
        assert coverage == pytest.approx(expected_static_coverage, abs=0.01)
    else:
        flow_motion, flow_bg = load_video_motion_and_background(
            video_path=vpath, n_frms=8, height=224, width=224, ablation="flow"
        )
        flow_coverage = (flow_bg.float().sum(0) > 0).float().mean().item()
        assert coverage == pytest.approx(flow_coverage, abs=0.02), (
            "random_mask 의 면적비가 flow 와 달라 용량 통제가 성립하지 않습니다."
        )
