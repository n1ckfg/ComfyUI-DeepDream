# ARCHITECTURE

PyTorch implementation of the [DeepDream algorithm](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html). DeepDream maximizes the activations of selected intermediate layers of a pretrained CNN via **gradient ascent on the input image**, producing psychedelic, dream-like imagery.

The repo is dual-mode: a standalone CLI/notebook project **and** a ComfyUI custom node pack (`__init__.py` registers the nodes). The same core (`deep_dream_static_image`) powers both.

Three output modes are supported:

| Mode | Script entry point | ComfyUI node | Node I/O |
|---|---|---|---|
| Static image | `deep_dream_static_image()` | `DeepDreamImage` | IMAGE (1 frame) → IMAGE (1 frame) |
| Ouroboros video | `deep_dream_video_ouroboros()` | `DeepDreamOuroboros` | IMAGE (seed frame) → IMAGE (batch of `ouroboros_length` frames) |
| DeepDream video | `deep_dream_video()` | `DeepDreamVideo` | IMAGE (batch of frames) → IMAGE (batch, temporally blended) |

## Repository layout

```
__init__.py             # ComfyUI node registration (DeepDreamImage / DeepDreamOuroboros / DeepDreamVideo)
deepdream.py            # Main algorithm + CLI entry point
playground.py           # Educational/experimental scripts (geometric transforms, blending, autograd, naive deepdream)
The Annotated DeepDream.ipynb   # Jupyter walkthrough of every concept (self-contained, no local imports)
models/
  definitions/          # Custom nn.Module wrappers around torchvision models, exposing intermediate layers
    vggs.py             # Vgg16, Vgg16Experimental
    resnets.py          # ResNet50
    googlenet.py        # GoogLeNet
    alexnet.py          # AlexNet
dd_utils/               # (was utils/, renamed to avoid sys.modules collisions in ComfyUI)
  constants.py          # Enums, ImageNet normalization, device, filesystem paths
  utils.py              # Image IO, tensor adapters, model factory, frame transforms, gaussian smoothing
  video_utils.py        # ffmpeg frame extraction / mp4 muxing, gif creation
data/
  input/                # Default sample inputs (images + mp4 clips)
  examples/             # Result gallery organized by experiment (pyramid_size, pyramid_ratio, datasets, ...)
  out-images/, out-videos/   # Created at import time; default dump targets
models/binaries/        # Created at import time; caches Places365 weights downloads
```

## Module responsibilities

### `deepdream.py` — core algorithm and CLI

- **`gradient_ascent(config, model, input_tensor, layer_ids_to_use, iteration)`** — one optimization step:
  1. Forward pass.
  2. MSE loss (reduction=mean) of each selected layer's activations against zeros.
  3. `loss.backward()` — because it's *ascent* (we maximize, not minimize), the gradient pushes activations up.
  4. Gradient post-processing: `CascadeGaussianSmoothing` (3 depthwise gaussian kernels, sigma grows with iteration), then mean/std normalization.
  5. Update: `input += lr * smooth_grad`, zero grads, clamp to normalized `[0,1]` image bounds (`LOWER/UPPER_IMAGE_BOUND` in constants, moved to the input tensor's device).
- **`deep_dream_static_image(config, img, model=None, device=None)`** — the workhorse used by all three script modes *and* all three ComfyUI nodes:
  - `model=None` → fetched via `fetch_and_prepare_model`; callers (e.g. the node layer) may inject a prebuilt, cached model. `device=None` → falls back to the global `DEVICE`; nodes pass the incoming IMAGE tensor's device so dreaming runs where ComfyUI put the pixels (CUDA / MPS / CPU).
  - Loads/normalizes input (ImageNet mean/std), optionally replaces with uniform noise.
  - **Image pyramid**: iterates `pyramid_size` levels from coarsest to finest; each level's size = base * `pyramid_ratio^(level - pyramid_size + 1)` (`dd_utils.get_new_shape`). Resizing the whole accumulated result per level (rather than blending octaves à la the original Caffe implementation) is a deliberate simplification that works well.
  - Per level: `num_gradient_ascent_iterations` steps, each bracketed by a **random circular spatial shift** (jitter) and its undo — `torch.roll` to decorrelate artifacts.
  - Returns channel-last, de-normalized, clipped `[0,1]` float32 numpy image.
- **`deep_dream_video_ouroboros(config)`** — dreams frame N, saves it, applies a geometric frame transform (`ZOOM`, `ZOOM_ROTATE`, `TRANSLATE` from `TRANSFORMS` enum, calibrated to 30 fps reference), feeds the transformed frame back as the next input. After `ouroboros_length` frames, muxes to mp4.
- **`deep_dream_video(config)`** — extracts frames with ffmpeg, then per frame: optionally **linear blend** with the previous *dreamed* frame (`blend` coefficient, 1.0 = current, 0.0 = previous) for temporal smoothness, deepdream it, save; finally muxes and cleans up temp dirs.
- **CLI (`__main__`)** — argparse with a deliberately small exposed surface. Config is a plain dict; `dump_dir` is derived from `data/out-videos` (ouroboros) or `data/out-images`, suffixed with `{model_name}_{pretrained_weights}`. Mode dispatch: `--create_ouroboros` → ouroboros, `.mp4` input → video, otherwise static image.

### `__init__.py` — ComfyUI node pack

ComfyUI executes `__init__.py` and consumes `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS` (category `DeepDream`). Three nodes wrap the same core, all returning ComfyUI `IMAGE` tensors `(B,H,W,C)` float32 `[0,1]` on the input's device:

| Node | Extra inputs beyond the shared dream params | Behavior |
|---|---|---|
| `DeepDreamImage` | — | dreams a single frame |
| `DeepDreamOuroboros` | `ouroboros_length` (INT), `fps` (INT), `frame_transform` (ZOOM / ZOOM_ROTATE / TRANSLATE) | dreams frame N, applies `dd_utils.transform_frame`, feeds it back as frame N+1's input; outputs an N-frame batch for a video-save node (e.g. VHS) |
| `DeepDreamVideo` | `blend` (FLOAT 0..1, default 0.85; 1.0 disables blending) | dreams each input frame, linearly blending with the previous *dreamed* frame (`dd_utils.linear_blend`) |

Shared dream params (built by `_dream_input_specs()`): `model_name`, `pretrained_weights`, `layers_to_use` (space-separated STRING, e.g. `"relu4_3 relu5_1"`), `pyramid_size`, `pyramid_ratio`, `num_gradient_ascent_iterations`, `lr`, `spatial_shift_size`, `smoothing_coefficient`, `target_width` (-1 keeps input width), `use_noise`, `seed` (noise only).

Key helpers:

- `_get_dream_model(model_name, weights, device)` — module-level cache of built dream models, keyed on `(model, weights, device)`, so repeated executions (and every frame of a video) reuse one model instead of rebuilding/downloading.
- `_validate_dream_params` / `_validate_layers` — fail early with a clear message for unsupported weights/model combos (PLACES_365 is ALEXNET/RESNET50-only) and for layer names not in the model's `layer_names`.
- `_build_config` — assembles the config dict the core expects (script-only keys like `dump_dir`/`should_display` are set to inert values).
- `_prepare_frame` / `_resize_to_width` / `_dream_frame` / `_frames_to_tensor` — tensor↔numpy boundary handling (ComfyUI IMAGE → HWC float32 `[0,1]` numpy → core → stacked batch tensor back on device), optional seeded noise, optional width resize.

Loading quirk: ComfyUI's loader does not add the node directory to `sys.path`, and the codebase uses bare top-level imports, so `__init__.py` starts with `sys.path.insert(0, <this dir>)`.

### `models/definitions/` — layer-exposing wrappers

All wrap torchvision pretrained models (`.eval()`, all params `requires_grad=False` to save memory) and replace the classification head with an **identity-like forward** that returns a `collections.namedtuple` of selected intermediate activations. Each exposes a `layer_names` list; `deep_dream.py` maps user-supplied layer names to tuple indices via `model.layer_names.index(name)` and fails gracefully with the list of valid names.

| Class | Exposed layers | Notes |
|---|---|---|
| `Vgg16` | `relu1_2, relu2_2, relu3_3, relu4_3` | 4 `Sequential` slices of `vgg16.features`; curated, pyramid-friendly |
| `Vgg16Experimental` | `relu3_3 ... relu5_3, mp5` | Every conv/relu/pool exposed as a named attr; deeper layers limit pyramid depth (low spatial res is the bottleneck) |
| `ResNet50` | `layer1, layer2, layer3, layer4` | Residual blocks unpacked down into individual conv/bn/relu of the last bottleneck (`layer42_*`) |
| `GoogLeNet` | `inception3b, inception4c, inception4d, inception4e` | Includes `transform_input` re-normalization quirk of torchvision |
| `AlexNet` | `relu1..relu5` | Supports **Places365** weights: downloads/caches `alexnet_places365.pth.tar` to `models/binaries/`, strips `.module` key prefixes, swaps final classifier to 365 classes |

`ResNet50` also supports Places365 (same download/cache pattern, key prefix stripping, 365-class head swap). `VGG16`, `Vgg16Experimental`, `GoogLeNet` are ImageNet-only.

### `dd_utils/utils.py`

- **Image IO**: `load_image` (cv2 BGR→RGB, optional width-preserving or exact resize, uint8→float32 `[0,1]`), `pre_process_numpy_img` / `post_process_numpy_img` (ImageNet normalize / denormalize + clip), `pytorch_input_adapter` (ToTensor, batch dim, `requires_grad=True`, to device) / `pytorch_output_adapter` (CPU, detach, CHW→HWC).
- **Model factory**: `fetch_and_prepare_model` maps `SupportedModels` enum → wrapper class.
- **`CascadeGaussianSmoothing`** (`nn.Module`): three depthwise gaussian convs (sigma multipliers 0.5/1.0/2.0), reflect padding, averaged — smooths gradients for more pleasing results. Kernels are built on the global `DEVICE` but moved to the input tensor's device in `forward` so the module works on whatever device ComfyUI uses.
- **`transform_frame`**: fps-scaled OpenCV affine warps for Ouroboros.
- **`get_new_shape`**: pyramid level sizing (exits with a hint if a level drops below 10 px).
- **`random_circular_spatial_shift`**: `torch.roll` jitter/undo.
- **Output plumbing**: `build_image_name` (parameter-transparent filenames encoding all hyperparameters), `save_and_maybe_display_image` (frames get zero-padded 6-digit names `000000.jpg`; static images get the full parameterized name), `parse_input_file` (absolute path or `data/input/` fallback), `linear_blend`, console header printers, `create_image_pyramid` (unused alternative).

### `dd_utils/video_utils.py` (script path only)

- `extract_frames` — ffmpeg dump to `frame_%06d.jpg`, returns fps metadata (fps read via `cv2.VideoCapture`).
- `create_video_from_intermediate_results` — ffmpeg mux of `%06d.jpg` frames (libx264, crf 25, yuv420p, even-dimension pad); frame count derived from dir listing or `ouroboros_length`.
- `create_gif` — imageio-based (used only from playground).
- `create_video_name` / `valid_frames` — naming + 6-digit `.jpg` pattern filtering.

### `dd_utils/constants.py`

- `IMAGENET_MEAN_1` / `IMAGENET_STD_1` (float32, for `[0,1]` images), `DEVICE` (CUDA if available else CPU), `LOWER/UPPER_IMAGE_BOUND` (clamping bounds in *normalized* space).
- Enums: `TRANSFORMS` (ZOOM, ZOOM_ROTATE, TRANSLATE), `SupportedModels`, `SupportedPretrainedWeights` (IMAGENET, PLACES_365).
- Filesystem constants; **side effect**: `os.makedirs` for `models/binaries`, `data/out-images`, `data/out-videos`, `data/out-videos/GIFS` at import.

### `playground.py` — educational

Standalone experiments selected via the `PLAYGROUND` enum: `understand_frame_transform` (matrix construction + OpenCV vs scipy perf note), `understand_blend`, `understand_pytorch_gradients` (why `act.backward(act)` ≡ MSE-sum/2), `deep_dream_simple` (15-line core algorithm without pyramid/jitter/smoothing), `create_gif`.

## Data flow (static image path)

```
input path
  └─ parse_input_file ─ load_image (BGR→RGB, [0,1] float32)
       └─ [optional uniform noise]
            └─ pre_process_numpy_img (ImageNet normalize)
                 └─ FOR each pyramid level (coarse → fine):
                      resize → pytorch_input_adapter ((1,3,H,W), requires_grad)
                      FOR each gradient-ascent iteration:
                        circular shift (+)
                        gradient_ascent: forward → MSE(activations, 0) → backward
                                         → cascade gaussian smooth → normalize
                                         → input += lr·grad, zero grad, clamp
                        circular shift (−)
                      pytorch_output_adapter → numpy
                 └─ post_process_numpy_img (denormalize, clip [0,1])
                      └─ save_and_maybe_display_image
```

The script's Ouroboros/video modes wrap `deep_dream_static_image` per frame, adding the feedback transform (Ouroboros) or previous-frame blending (video), then mux via `dd_utils.video_utils.create_video_from_intermediate_results` (ffmpeg). The ComfyUI nodes do the same per-frame wrapping but skip all file IO — they return frame batches as IMAGE tensors, which the user wires to any video-save node.

## Key invariants / conventions

- **Image representation**: channel-last RGB `np.ndarray`, float32 in `[0,1]` at the numpy boundary; `(1,3,H,W)` PyTorch tensors in normalized space internally. In ComfyUI the external boundary is an IMAGE tensor `(B,H,W,C)` float32 `[0,1]`; nodes convert at the edge and pass `device=tensor.device` into the core, so all device-dependent pieces (clamping bounds, smoothing kernels) move to the input's device rather than trusting the import-time `DEVICE`.
- **The helper package is named `dd_utils`, not `utils`** — the generic name collides with other ComfyUI custom nodes' `utils` packages in `sys.modules` (e.g. Civicomfy), which breaks bare `import utils.utils` at node load time.
- **Frames are named `%06d.jpg`** (script path) — the ffmpeg muxer and `valid_frames` both depend on this.
- **Layer selection is by name** (`--layers_to_use relu4_3` CLI / `layers_to_use` node input); the name→index resolution lives in `deep_dream_static_image`, and each model wrapper owns its own `layer_names`.
- **Model weights are frozen** everywhere (`requires_grad=False`); only the input tensor is optimized.
- **Dream models are cached** per `(model_name, weights, device)` in `__init__.py`; the script path builds a fresh model per run.
- **Places365 weights are downloaded lazily** and cached under `models/binaries/`.
- Output filenames encode the full hyperparameter set for reproducibility.
- The code is written for readability/education: heuristics are inline-commented ("magic number" 9 kernel, arbitrary sigma schedule, fps-calibrated transforms), and `playground.py` + the notebook decompose each concept.

## Dependencies

`environment.yml` pins: python with `torch`, `torchvision`, `numpy`, `scipy`, `opencv-python`, `imageio`, `matplotlib`, plus system `ffmpeg` (required at runtime for any video mode; checked via `shutil.which`).
