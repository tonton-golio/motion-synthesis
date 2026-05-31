<div align="center">

# 🕺 Rapid Motion Synthesis · **RaMoS**

**Text-to-motion by reverse diffusion in a learned latent space** — diverse 3D human motion from short prompts, with modest compute.

PyTorch · PyTorch Lightning · TensorBoard

</div>

---

## What it is

RaMoS generates 3D human motion in **two stages**, the standard *latent diffusion* recipe (Rombach et al.; MLD, Chen et al. 2023) applied to motion:

1. **Compression** — a transformer **VAE** squeezes a motion clip `(T=160 frames × 22 joints × 3)` into a small latent `z ∈ ℝ²⁵⁶`.
2. **Generation** — a **conditional diffusion model** learns to denoise in that latent space, steered by a CLIP text embedding. Sampled latents are handed back to the VAE decoder to become motion.

Diffusing in the compact latent (rather than on raw joints) is what keeps training and sampling cheap.

```
                          ┌─────────────────────── Stage 1: Motion VAE ───────────────────────┐
   motion (B,160,22,3) ──▶│ skel-embed ▶ +dist-tokens ▶ Transformer enc ▶  μ,logσ²  ▶  z (B,1,256) │
                          │                                                          │            │
                          │            motion (B,160,22,3) ◀ Linear ◀ Transformer dec ◀ query+z    │
                          └────────────────────────────────────────────────────────────────────┘
                                                          ▲ z
                          ┌──────────────────── Stage 2: Latent Diffusion ───────────────────┐
   "a person walks" ─CLIP─▶ cond ─┐                                                           │
                                   ├─▶ ε-predictor ◀── x_t ◀── q(x_t|z)         (train)        │
            z₀ ~ VAE latents ──────┘                                                           │
                                                                                              │
            N(0,I) ─▶ DDIM reverse (+classifier-free guidance) ─▶ ẑ ─▶ VAE.decode  (sample)   │
                          └──────────────────────────────────────────────────────────────────┘
```

The repo also contains a **MNIST latent-diffusion** pipeline (a clean reference), a **single-frame Pose VAE** (incl. a graph-NN variant), and the **AMASS / HumanML3D** preprocessing + analysis notebooks.

---

## Architecture at a glance

| Stage | Model | In → Out | Key idea |
|------|-------|----------|----------|
| Compression | `MotionVAE_MLD` | `(B,160,22,3)` → `z (B,1,256)` | ACTOR/MLD learnable **distribution tokens**; decoder **cross-attends** `T` positional queries to `z`. The batch axis is never reshaped, so samples can't leak into each other. |
| Loss | `MotionVAELoss` | — | position + velocity + foot-contact (SmoothL1, masked by true length) + **batch-invariant KL** with warm-up. |
| Generation | `MotionLatentDiffusion` | `text → z` | **ε-prediction**, cosine schedule, **DDPM/DDIM** sampling, **classifier-free guidance**. |

---

## Quickstart

```bash
# 1. Environment (uv recommended; Python 3.13)
uv venv --python 3.13 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

# 2. Run the test suite (CPU, synthetic data — no dataset needed)
.venv/bin/python -m pytest motion_latent_diffusion/tests -q

# 3. Smoke-test the VAE forward pass
.venv/bin/python -m motion_latent_diffusion.modules.MotionVAE --seq_len 160

# 4. Train (needs the HumanML3D data — see "Data" below)
python motion_latent_diffusion/main.py --model_name VAEMLD     --mode train   # stage 1: VAE
python motion_latent_diffusion/main.py --model_name LD_VAEMLD  --mode train   # stage 2: diffusion
```

Configs live in `motion_latent_diffusion/configs/` (`config_motion_VAE.yaml`, `config_motion_LD.yaml`). TensorBoard logs and checkpoints are written under `motion_latent_diffusion/logs/`.

---

## Verification

Correctness is pinned by a fast, **dataset-free** CPU test suite — it runs on synthetic motion and a toy 2-D distribution, so it verifies the *math and wiring* without HumanML3D or a GPU:

| Test | Guards |
|------|--------|
| `test_per_sample_independence` | the latent of one sample never depends on its batch neighbours |
| `test_kl_batch_invariance` | KL is per-sample and batch-size invariant |
| `test_overfit_one_batch` | the VAE actually learns; KL stays finite and non-collapsed |
| `test_lengths_masking` | padded frames are excluded from the loss |
| `test_diffusion_toy_2d` | DDPM/DDIM recovers a known distribution; CFG steers; noise is unit-variance |
| `test_eval_and_latentdiff` | MPJPE / PA-MPJPE / foot-skate metrics; conditional generation separates |

```bash
.venv/bin/python -m pytest motion_latent_diffusion/tests -q   # 24 passed
```

> **Status.** The pipeline is architecturally complete and verified on the tests above. End-to-end *quality* numbers (FID, R-precision, MPJPE on held-out motion) require training on the full HumanML3D dataset, which is not bundled here.

---

## Repository layout

```
motion_latent_diffusion/
  modules/
    motion_vae_mld.py        # the Motion VAE (compression)
    MotionVAE.py             # LightningModule + legacy VAE variants
    MotionLatentDiffusion.py # latent diffusion LightningModule (ε-pred + CFG)
    diffusion.py             # model-agnostic GaussianDiffusion (DDPM/DDIM)
    loss.py                  # batch-invariant KL + masked 4-term motion loss
    preprocessing.py         # root-removal, heading canon, normalize, pad
    eval_metrics.py          # MPJPE, PA-MPJPE, velocity, foot-skate
    MotionData.py            # HumanML3D clip dataset + DataModule
    LatentMotionData.py      # latent dataset for stage 2
    PoseVAE.py / PoseData.py # single-frame pose VAE
  configs/                   # YAML configs
  scripts/                   # training entrypoints
  tests/                     # CPU synthetic test suite
  data/                      # AMASS/HumanML3D preprocessing notebooks
mnist_latent_diffusion/      # reference latent-diffusion pipeline on MNIST
app/                         # Streamlit explorer (theory pages + inference)
```

---

## Data

Motion comes from **AMASS**, processed to the **HumanML3D** layout (22 joints, `recover_from_ric`). Stage 1 here consumes raw joint positions `(T, 22, 3)`; preprocessing removes global root translation, canonicalizes heading, normalizes per-axis, and zero-pads to a fixed length with a true-length mask. Place the dataset under the paths in `config_motion_VAE.yaml` (`stranger_repos/HumanML3D/...`) or edit them to your location.

---

## References

- **MLD** — Chen et al., *Executing your Commands via Motion Diffusion in Latent Space*, CVPR 2023
- **ACTOR** — Petrovich et al., *Action-Conditioned 3D Human Motion Synthesis with Transformer VAE*, ICCV 2021
- **HumanML3D** — Guo et al., *Generating Diverse and Natural 3D Human Motions from Text*, CVPR 2022
- **DDPM / DDIM** — Ho et al. 2020; Song et al. 2021 · **Classifier-Free Guidance** — Ho & Salimans 2022

<div align="center"><sub>Built as a study in compact, controllable motion generation.</sub></div>
