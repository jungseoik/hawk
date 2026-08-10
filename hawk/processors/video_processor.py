"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from hawk.common.registry import registry
from decord import VideoReader
import decord
import numpy as np
from hawk.processors import transforms_video
from hawk.processors.base_processor import BaseProcessor
from hawk.processors.randaugment import VideoRandomAugment
from hawk.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd
import cv2  # Import OpenCV

MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")

mag_threshold = 0.2

def _block_shuffle_mask(mask, block=16, seed=None):
    """마스크를 블록 단위로 뒤섞는다 (면적비 보존, 위치 정보 파괴).

    화소 단위로 섞으면 소금-후추 잡음이 되어 인코더가 의미와 무관한 이유로 다르게
    반응한다. 블록 단위로 섞으면 국소 질감은 남기고 "어느 영역이 움직였는가"라는
    의미적 대응만 끊을 수 있다.
    """
    h, w = mask.shape
    ph, pw = (block - h % block) % block, (block - w % block) % block
    padded = np.pad(mask, ((0, ph), (0, pw)))
    H, W = padded.shape

    blocks = padded.reshape(H // block, block, W // block, block).transpose(0, 2, 1, 3)
    flat = blocks.reshape(-1, block, block)

    rng = np.random.default_rng(seed)
    flat = flat[rng.permutation(len(flat))]

    out = flat.reshape(H // block, W // block, block, block).transpose(0, 2, 1, 3).reshape(H, W)
    return out[:h, :w]


def compute_motion_and_background(frames, frame_list, ablation="flow"):
    """Compute both motion and background frames in a single optical flow pass.

    Motion mask: regions where optical flow magnitude > threshold (moving objects)
    Background mask: inverse of motion mask (static scene context)
    """
    motion_frames = []
    background_frames = []
    numpy_frame = frames.asnumpy()
    # 주의: 이전 구현은 `frame_list[0] = 1`로 첫 인덱스를 덮어썼다. i-1 = -1 로 뒤쪽
    # 프레임을 참조하는 것을 막으려는 의도였겠지만, 그 결과 동적/정적 스트림의 첫
    # 프레임만 외형 스트림과 다른 시점이 되어 상보성 항등식이 그 프레임에서 깨졌다
    # (실측: frame 0 최대 오차 18, 나머지 프레임 0.000). 호출자의 리스트를 제자리에서
    # 변형해 타임스탬프 메시지도 어긋났다. 출력 프레임은 그대로 두고 "이전 프레임"만
    # 클램프한다 — i = 0 이면 플로우가 0이 되어 전 화면이 정적 스트림으로 간다.
    for i in frame_list:
        prev_frame = cv2.cvtColor(numpy_frame[max(i - 1, 0)], cv2.COLOR_RGB2GRAY)
        current_frame = cv2.cvtColor(numpy_frame[i], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_frame, current_frame, None, 0.5, 3, 10, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])

        motion_mask = (mag > mag_threshold).astype(np.uint8)

        # --- 정적 스트림 통제군 (ablation) ---------------------------------
        # 네 조건 모두 아키텍처·파라미터 수·시각 토큰 수가 동일하고, 정적 스트림이
        # 담는 *내용*만 다르다. 따라서 성능 차이는 용량이 아니라 내용에 귀속된다.
        #   flow        : 제안 방식. 정적 = (1 − M) ⊙ x
        #   random_mask : M을 블록 단위로 뒤섞어 면적비는 보존하되 위치를 무의미하게.
        #                 여기서도 이득이 나오면 "배경 내용" 주장은 성립하지 않는다.
        #   duplicate   : 정적 = 원본 프레임 전체. 분해 없이 스트림만 하나 더.
        #   zero        : 정적 = 0. 용량은 있고 내용은 없는 조건.
        if ablation == "random_mask":
            bg_mask = 1 - _block_shuffle_mask(motion_mask)
        elif ablation == "duplicate":
            bg_mask = np.ones_like(motion_mask)
        elif ablation == "zero":
            bg_mask = np.zeros_like(motion_mask)
        elif ablation == "flow":
            bg_mask = 1 - motion_mask
        else:
            raise ValueError(f"unknown static-stream ablation: {ablation!r}")

        motion_mask_3ch = np.stack((motion_mask,) * 3, axis=-1)
        motion_frames.append(numpy_frame[i] * motion_mask_3ch)

        bg_mask_3ch = np.stack((bg_mask,) * 3, axis=-1)
        background_frames.append(numpy_frame[i] * bg_mask_3ch)

    return np.stack(motion_frames, axis=0), np.stack(background_frames, axis=0)


def compute_optical_flow(frames, frame_list):
    """Compute motion frames only (kept for backward compatibility)."""
    motion_frames, _ = compute_motion_and_background(frames, frame_list)
    return motion_frames


def compute_background(frames, frame_list):
    """Compute background frames only (kept for backward compatibility)."""
    _, background_frames = compute_motion_and_background(frames, frame_list)
    return background_frames

def flow_to_color(flow):
    # 将光流转换为可视化的颜色图像
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow_vis

def load_video_motion(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge('native')
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    # temp_frms = vr.get_batch(indices)
    frames = vr.get_batch(np.arange(len(vr)))

    # print(type(frames))
    temp_frms = compute_optical_flow(frames,indices)
    
    decord.bridge.set_bridge("torch")
    
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


def load_video_background(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge('native')
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    frames = vr.get_batch(np.arange(len(vr)))

    temp_frms = compute_background(frames, indices)

    decord.bridge.set_bridge("torch")

    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


def load_video_motion_and_background(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg=False, ablation="flow"):
    """Load video and compute both motion and background frames in a single optical flow pass."""
    decord.bridge.set_bridge('native')
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    frames = vr.get_batch(np.arange(len(vr)))
    motion_frms, bg_frms = compute_motion_and_background(frames, indices, ablation=ablation)

    decord.bridge.set_bridge("torch")

    motion_tensor = torch.from_numpy(motion_frms).permute(3, 0, 1, 2).float()
    bg_tensor = torch.from_numpy(bg_frms).permute(3, 0, 1, 2).float()

    if not return_msg:
        return motion_tensor, bg_tensor

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return motion_tensor, bg_tensor, msg


def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    # print(type(temp_frms))
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


# ---------------------------------------------------------------------------
# 세 스트림(외형 / 동적 / 정적)의 정합을 보장하는 헬퍼
#
# CERBERUS의 최상위 주장은 상보성 항등식  M ⊙ x + (1 − M) ⊙ x = x  이다. 이 등식은
# 세 스트림이 **같은 프레임을 같은 방식으로 자른** 것일 때만 성립한다. 수정 전 코드는
# 두 지점에서 이를 깨뜨렸다.
#
#  1) 프레임 선택: load_video 와 load_video_motion_and_background 를 각각 호출하는데,
#     "headtail" 샘플링은 rnd.sample 기반이라 호출마다 다른 프레임이 뽑힌다.
#     → 외형 스트림과 동적/정적 스트림이 서로 다른 시점을 보게 된다.
#  2) 증강: transform 안의 RandomResizedCropVideo 는 호출마다 새 crop 파라미터를
#     뽑으므로, 세 번 호출하면 스트림마다 다른 영역이 잘린다.
#     (실측: (0,80,360,480) / (31,120,320,394) / (5,71,343,350))
#
# 두 헬퍼는 각 난수 소비 직전에 동일한 시드를 걸어 세 스트림을 일치시킨다. 샘플 간
# 다양성은 유지된다 — 시드 자체를 매 샘플 새로 뽑기 때문이다.
# ---------------------------------------------------------------------------


def load_streams_aligned(vpath, n_frms, image_size, sampling="uniform", ablation="flow",
                         return_msg=False):
    """세 스트림을 동일한 프레임 인덱스에서 만든다.

    sampling="uniform"이면 선택이 결정론적이라 시드가 없어도 일치하지만, headtail과
    같은 난수 기반 샘플링에서도 동일하게 동작하도록 항상 시드를 맞춘다.
    """
    seed = rnd.randint(0, 2**32 - 1)

    rnd.seed(seed)
    loaded = load_video(
        video_path=vpath, n_frms=n_frms, height=image_size, width=image_size,
        sampling=sampling, return_msg=return_msg,
    )
    clip, msg = loaded if return_msg else (loaded, None)

    rnd.seed(seed)
    clip_motion, clip_background = load_video_motion_and_background(
        video_path=vpath, n_frms=n_frms, height=image_size, width=image_size,
        sampling=sampling, ablation=ablation,
    )

    if return_msg:
        return clip, clip_motion, clip_background, msg
    return clip, clip_motion, clip_background


def apply_shared_transform(transform, clips):
    """세 스트림에 동일한 crop 파라미터로 transform을 적용한다.

    RandomResizedCrop.get_params 는 torch RNG를 쓰므로, 호출 직전 같은 시드를 걸면
    같은 (i, j, h, w)가 나온다.

    전역 RNG 상태는 반드시 복원한다. `torch.manual_seed` 는 CPU 뿐 아니라 모든 CUDA
    디바이스의 RNG를 함께 시드하므로, 복원하지 않으면 DataLoader 워커 밖에서 이 함수를
    부르는 경로(app.py 데모, 평가 스크립트)에서 **생성 RNG가 매 비디오 로드마다
    크롭 시드로 리셋된다.** 그렇게 되면 생성 결과가 입력 파이프라인의 함수가 되어,
    같은 입력을 반복 생성해 얻는 BSI 의 자기 일치도 기준선 자체가 오염된다.
    """
    seed = int(torch.randint(0, 2**31 - 1, (1,)).item())

    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        out = []
        for clip in clips:
            torch.manual_seed(seed)
            out.append(transform(clip))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return out


class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        # clamp 없이 캐스트하면 PyTorch 는 클램프가 아니라 **wraparound(mod 256)** 한다:
        #   [-3.7, -1.2, 258.9, 300.0] -> [253, 255, 2, 44]
        # bicubic 보간은 마스크 경계(= 움직이는 객체의 윤곽)에서 값을 범위 밖으로
        # 밀어내므로, 검게 가려져야 할 화소가 253 같은 밝은 점으로 뒤집힌다.
        # 실측 범위 이탈 비율: appearance 0.000%, motion 0.52%, background 0.36%
        # — 마스킹된 스트림에서만 발생한다. 반드시 클램프한 뒤 캐스트한다.
        return tensor.clamp(0, 255).to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                # self.normalize,
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clips = load_streams_aligned(
            vpath, self.n_frms, self.image_size, sampling="headtail"
        )
        return tuple(apply_shared_transform(self.transform, clips))

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                # self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        # 평가 경로는 결정론적이어야 하므로 uniform으로 통일한다. (수정 전에는 외형만
        # uniform, 동적/정적은 headtail이라 스트림 간 프레임이 어긋났다.)
        clips = load_streams_aligned(
            vpath, self.n_frms, self.image_size, sampling="uniform"
        )
        return tuple(apply_shared_transform(self.transform, clips))

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)
