# SFG
### Training-free guidance for multimodal diffusion transformers

SFG provides inference-time guidance for image and video generation, without
fine-tuning or additional learned weights.

**Supported models:** SD3 Medium · SD3.5 Medium · FLUX.1-dev ·
FLUX-de-distill · HunyuanVideo-1.5 (T2V / I2V)

## Installation

Use Python 3.10+ and a CUDA GPU.

```bash
git clone https://github.com/yyc1215lll-creator/SFG.git
cd SFG

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-image.txt
```

For video generation, use a separate environment:

```bash
pip install -r video/third_party/HunyuanVideo-1.5/requirements.txt
pip install flash-attn==2.8.3 --no-build-isolation
```

Install a PyTorch build compatible with your CUDA environment before installing
FlashAttention.

## Prepare model weights

Download the model and pass its local directory with `--weights`.

| Model | Weights |
|---|---|
| SD3 Medium | [stabilityai/stable-diffusion-3-medium-diffusers](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) |
| SD3.5 Medium | [stabilityai/stable-diffusion-3.5-medium](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium) |
| FLUX.1-dev | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) |
| FLUX-de-distill transformer | [InstantX/flux-dev-de-distill-diffusers](https://huggingface.co/InstantX/flux-dev-de-distill-diffusers) |
| HunyuanVideo-1.5 | [Upstream model setup](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5) |

Use complete Diffusers model directories for image models. FLUX-de-distill
additionally requires a de-distilled transformer directory with
`guidance_embeds=false`; pass it with `--transformer`.
For HunyuanVideo, use the native Base `480p_t2v` or `480p_i2v` weights.

For example, after accepting any required model access terms:

```bash
hf auth login
hf download stabilityai/stable-diffusion-3.5-medium --local-dir models/sd35m
hf download black-forest-labs/FLUX.1-dev --local-dir models/flux-dev
hf download InstantX/flux-dev-de-distill-diffusers --local-dir models/flux-de-distill
hf download tencent/HunyuanVideo-1.5 --local-dir models/HunyuanVideo-1.5
```

The downloaded `models/flux-de-distill` directory itself is the transformer
directory to pass to `--transformer`.

## Image generation

Provide a text file with one prompt per line. Example prompts are included in
[examples/prompts.txt](examples/prompts.txt).

```bash
# SD3.5 Medium
python run_image.py --model sd35m --weights /path/to/SD3.5-medium \
  --prompt-file examples/prompts.txt --output outputs/sd35m

# SD3 Medium
python run_image.py --model sd3m --weights /path/to/SD3-medium \
  --prompt-file examples/prompts.txt --output outputs/sd3m

# FLUX.1-dev
python run_image.py --model flux-dev --weights /path/to/FLUX.1-dev \
  --prompt-file examples/prompts.txt --output outputs/flux-dev

# FLUX-de-distill
python run_image.py --model flux-de-distill --weights /path/to/FLUX.1-dev \
  --transformer /path/to/de-distilled/transformer \
  --prompt-file examples/prompts.txt --output outputs/flux-de-distill
```

Images are saved to `OUTPUT/result/`. Use a fresh output directory for each run.
The default batch size is 4; use `--batch-size 1` to reduce memory usage.
The seed for each image is `--seed + image index`.

## Video generation

Create a prompt manifest, then launch generation:

```bash
# Text-to-video
python prepare_video_manifest.py \
  --prompt 'A boat moves across a calm lake.' --output t2v.jsonl

torchrun --standalone --nproc_per_node=1 run_video.py --task t2v \
  --weights /path/to/HunyuanVideo-1.5 \
  --manifest t2v.jsonl --output outputs/t2v

# Image-to-video
python prepare_video_manifest.py \
  --prompt 'The subject slowly turns its head.' \
  --reference-image /path/to/reference.png --output i2v.jsonl

torchrun --standalone --nproc_per_node=1 run_video.py --task i2v \
  --weights /path/to/HunyuanVideo-1.5 \
  --manifest i2v.jsonl --output outputs/i2v

# Image-to-video with text-only guidance
torchrun --standalone --nproc_per_node=1 run_video.py --task i2v-text \
  --weights /path/to/HunyuanVideo-1.5 \
  --manifest i2v.jsonl --output outputs/i2v-text
```

Use `--offloading` to enable model offloading or `--resume` to resume a run.

## Configuration

Model-specific defaults are provided in [defaults.py](defaults.py).
Standard generation does not require manually setting guidance parameters.

| Model / task | CFG / g | uS | uX | w | Steps | SFG window |
|---|---:|---:|---:|---:|---:|---|
| SD3M | 1 | +0.15 | −0.40 | 6 | 40 | 1–40 |
| SD3.5M | 7.5 | +0.25 | −0.25 | 3.5 | 40 | 1–20 |
| FLUX-dev | 1 | +0.35 | −0.35 | 14 | 28 | 1–28 |
| FLUX-de-distill | 3.5 | +0.35 | −0.45 | 7.5 | 28 | 1–28 |
| T2V | 1 | +0.25 | −0.25 | 4 | 25 | 1–7 |
| I2V | 1 | +0.275 | −0.275 | 6 | 25 | 1–7 |
| I2V text-only | 1 | +0.20 | −0.20 | 3 | 25 | 1–7 |

Image defaults use BF16, 1024×1024 resolution, and seed 42.
Video defaults use BF16, 848×480 resolution, 121 frames, and 24 FPS.

## Multi-GPU generation

For image generation, replace `python` with
`torchrun --standalone --nproc_per_node=8`.
For video generation, set `--nproc_per_node=8` with a multi-row manifest.
Each GPU loads a complete model and processes independent samples.

## Acknowledgments

Built on [Diffusers](https://github.com/huggingface/diffusers),
[Stable Diffusion](https://github.com/Stability-AI),
[FLUX](https://github.com/black-forest-labs/flux), and
[HunyuanVideo-1.5](https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5).
Third-party code and model weights retain their respective licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
