import contextlib
import os
import sys

# ComfyUI loads this package without adding its directory to sys.path,
# and the existing codebase uses bare top-level imports (utils, models, deepdream).
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import numpy as np
import cv2 as cv
import torch

import deepdream
import dd_utils.utils as utils
from dd_utils.constants import SupportedModels, SupportedPretrainedWeights, TRANSFORMS


#
# Shared helpers
#

_dream_model_cache = {}


@contextlib.contextmanager
def _autograd_enabled():
    """Re-enable autograd inside ComfyUI's torch.inference_mode() execution context.

    ComfyUI runs every node under `torch.inference_mode()` (execution.py), which is
    strictly stronger than `no_grad()`: tensors *created* there are "inference tensors"
    that can never take part in autograd. DeepDream is gradient ascent on the input, so
    it needs a real graph, hence `inference_mode(False)`.

    Everything autograd touches must be built inside this block, model weights included -
    conv2d saves its weight for the backward pass, and an inference-tensor weight raises
    "Inference tensors cannot be saved for backward" even though the weights are frozen.
    """
    with torch.inference_mode(False), torch.enable_grad():
        yield


def _get_dream_model(model_name, pretrained_weights, device):
    key = (model_name, pretrained_weights, str(device))
    if key not in _dream_model_cache:
        _dream_model_cache[key] = utils.fetch_and_prepare_model(model_name, pretrained_weights, device)
    return _dream_model_cache[key]


def _validate_dream_params(model_name, pretrained_weights, layers_to_use):
    if pretrained_weights == SupportedPretrainedWeights.PLACES_365.name \
            and model_name not in (SupportedModels.ALEXNET.name, SupportedModels.RESNET50.name):
        raise ValueError(f'PLACES_365 weights are only supported for ALEXNET and RESNET50, got model {model_name}.')


def _validate_layers(model, layers_to_use):
    invalid = [l for l in layers_to_use if l not in model.layer_names]
    if invalid:
        raise ValueError(f'Invalid layer names {invalid} for model {type(model).__name__}. '
                         f'Available layers: {model.layer_names}')


def _build_config(model_name, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
                  num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient):
    return {
        'model_name': model_name,
        'pretrained_weights': pretrained_weights,
        'layers_to_use': [l for l in layers_to_use.split() if l],
        'pyramid_size': pyramid_size,
        'pyramid_ratio': pyramid_ratio,
        'num_gradient_ascent_iterations': num_gradient_ascent_iterations,
        'lr': lr,
        'spatial_shift_size': spatial_shift_size,
        'smoothing_coefficient': smoothing_coefficient,
        'img_width': -1,
        'use_noise': False,
        'should_display': False,
        'fps': 30,
        'frame_transform': TRANSFORMS.ZOOM_ROTATE.name,
    }


def _resize_to_width(frame, target_width):
    if target_width is None or target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w == target_width:
        return frame
    new_height = int(h * (target_width / w))
    return cv.resize(frame, (target_width, new_height), interpolation=cv.INTER_CUBIC)


def _prepare_frame(image, use_noise, seed, target_width):
    frame = image[0].to('cpu').detach().numpy().astype(np.float32)
    if use_noise:
        rng = np.random.RandomState(seed % (2**32))
        frame = rng.uniform(low=0.0, high=1.0, size=frame.shape).astype(np.float32)
    return _resize_to_width(frame, target_width)


def _dream_frame(config, model, device, frame):
    return deepdream.deep_dream_static_image(config, frame, model=model, device=device)


def _frames_to_tensor(frames, device):
    return torch.from_numpy(np.stack(frames)).to(device)


def _dream_input_specs():
    return {
        "model_name": ([m.name for m in SupportedModels],
                       {"default": SupportedModels.VGG16_EXPERIMENTAL.name}),
        "pretrained_weights": ([w.name for w in SupportedPretrainedWeights],
                               {"default": SupportedPretrainedWeights.IMAGENET.name}),
        "layers_to_use": ("STRING", {"default": "relu4_3"}),
        "pyramid_size": ("INT", {"default": 4, "min": 1, "max": 16, "step": 1}),
        "pyramid_ratio": ("FLOAT", {"default": 1.8, "min": 1.0, "max": 3.0, "step": 0.05}),
        "num_gradient_ascent_iterations": ("INT", {"default": 10, "min": 1, "max": 100, "step": 1}),
        "lr": ("FLOAT", {"default": 0.09, "min": 0.0, "max": 1.0, "step": 0.01}),
        "spatial_shift_size": ("INT", {"default": 32, "min": 0, "max": 512, "step": 1}),
        "smoothing_coefficient": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.1}),
        "target_width": ("INT", {"default": -1, "min": -1, "max": 8192, "step": 1}),
        "use_noise": ("BOOLEAN", {"default": False}),
        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "step": 1}),
    }


class DeepDreamImage:
    """Applies the DeepDream algorithm (gradient ascent on a pretrained CNN) to a single image."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {"image": ("IMAGE",)}
        inputs.update(_dream_input_specs())
        return {"required": inputs}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dream"
    CATEGORY = "DeepDream"

    def dream(self, image, model_name, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
              num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient,
              target_width, use_noise, seed):
        device = image.device
        _validate_dream_params(model_name, pretrained_weights, layers_to_use)
        config = _build_config(model_name, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
                               num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient)
        with _autograd_enabled():
            model = _get_dream_model(model_name, pretrained_weights, device)
            _validate_layers(model, config['layers_to_use'])

            frame = _prepare_frame(image, use_noise, seed, target_width)
            dreamed = _dream_frame(config, model, device, frame)
            return (torch.from_numpy(dreamed).to(device).unsqueeze(0),)


class DeepDreamOuroboros:
    """Dreams a seed frame, feeds the (optionally transformed) output back as the next input, and repeats.
    Outputs one batch per frame, ready for a video save node."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "image": ("IMAGE",),
            "ouroboros_length": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1}),
            "fps": ("INT", {"default": 30, "min": 1, "max": 240, "step": 1}),
            "frame_transform": ([t.name for t in TRANSFORMS], {"default": TRANSFORMS.ZOOM_ROTATE.name}),
        }
        inputs.update(_dream_input_specs())
        return {"required": inputs}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dream"
    CATEGORY = "DeepDream"

    def dream(self, image, ouroboros_length, fps, frame_transform, model_name, pretrained_weights, layers_to_use,
              pyramid_size, pyramid_ratio, num_gradient_ascent_iterations, lr, spatial_shift_size,
              smoothing_coefficient, target_width, use_noise, seed):
        device = image.device
        _validate_dream_params(model_name, pretrained_weights, layers_to_use)
        config = _build_config(model_name, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
                               num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient)
        config['fps'] = fps
        config['frame_transform'] = frame_transform
        config['ouroboros_length'] = ouroboros_length
        with _autograd_enabled():
            model = _get_dream_model(model_name, pretrained_weights, device)
            _validate_layers(model, config['layers_to_use'])

            frame = _prepare_frame(image, use_noise, seed, target_width)
            frames = []
            for _ in range(ouroboros_length):
                frame = _dream_frame(config, model, device, frame)
                frames.append(frame)
                frame = utils.transform_frame(config, frame)
            return (_frames_to_tensor(frames, device),)


class DeepDreamVideo:
    """DeepDreams every frame of an input video (batch of frames).
    Each frame is optionally linearly blended with the previous dreamed frame for temporal smoothness
    (blend=1.0 disables blending, 0.0 keeps the previous frame)."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "image": ("IMAGE",),
            "blend": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
        }
        inputs.update(_dream_input_specs())
        return {"required": inputs}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "dream"
    CATEGORY = "DeepDream"

    def dream(self, image, blend, model_name, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
              num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient,
              target_width, use_noise, seed):
        device = image.device
        _validate_dream_params(model_name, pretrained_weights, layers_to_use)
        config = _build_config(model_name, pretrained_weights, layers_to_use, pyramid_size, pyramid_ratio,
                               num_gradient_ascent_iterations, lr, spatial_shift_size, smoothing_coefficient)
        with _autograd_enabled():
            model = _get_dream_model(model_name, pretrained_weights, device)
            _validate_layers(model, config['layers_to_use'])

            frames = image.to('cpu').detach().numpy().astype(np.float32)
            last_dreamed = None
            dreamed_frames = []
            for i in range(frames.shape[0]):
                frame = frames[i]
                if use_noise:
                    rng = np.random.RandomState((seed + i) % (2**32))
                    frame = rng.uniform(low=0.0, high=1.0, size=frame.shape).astype(np.float32)
                frame = _resize_to_width(frame, target_width)
                if last_dreamed is not None:
                    frame = utils.linear_blend(last_dreamed, frame, blend)
                dreamed_frame = _dream_frame(config, model, device, frame)
                last_dreamed = dreamed_frame
                dreamed_frames.append(dreamed_frame)
            return (_frames_to_tensor(dreamed_frames, device),)


NODE_CLASS_MAPPINGS = {
    "DeepDreamImage": DeepDreamImage,
    "DeepDreamOuroboros": DeepDreamOuroboros,
    "DeepDreamVideo": DeepDreamVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DeepDreamImage": "DeepDream Image",
    "DeepDreamOuroboros": "DeepDream Ouroboros",
    "DeepDreamVideo": "DeepDream Video",
}
