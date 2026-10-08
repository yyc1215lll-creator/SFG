# Third-party notices

SFG builds on the projects below. Their source headers and licenses are retained.

- HunyuanVideo-1.5 source is vendored under `video/third_party/HunyuanVideo-1.5`.
  Its upstream source revision is `60783e704160023913bee78f0b47036d393d4dfa`.
  Preserve the included `LICENSE`, `NOTICE`, and original source headers.
  `hyvideo/pipelines/hunyuan_video_pipeline.py` includes CFG window support.
- The FLUX de-distill pipeline derives from Hugging Face Diffusers; its Apache
  2.0 copyright/license header remains in `image/pipeline_flux_de_distill.py`.
- Optional video guidance modules include CFG-Zero* and STGuidance-derived code;
  their upstream references are retained in the source files.
- Model weights, model license grants, benchmark assets, evaluator weights and
  their licenses are not bundled. Obtain these separately from their providers.

Third-party licenses continue to apply to their respective code and models.
