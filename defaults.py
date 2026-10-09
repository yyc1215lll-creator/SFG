"""Default SFG settings. All layer selections default to 'all'.

u_s and u_x control the two cross-stream attention contributions.
omega scales the clean-minus-counterfactual prediction residual.
Windows are 1-based inclusive; end_step=0 means the final step.
"""

IMAGE_DEFAULTS = {
    "sd3m": dict(steps=40, guidance=1.0, u_s=0.15, u_x=-0.40,
                 omega=6.0, start_step=1, end_step=0),
    "sd35m": dict(steps=40, guidance=7.5, u_s=0.25, u_x=-0.25,
                  omega=3.5, start_step=1, end_step=20),
    # Original FLUX-dev uses its embedded guidance input.
    "flux-dev": dict(steps=28, guidance=1.0, u_s=0.35, u_x=-0.35,
                     omega=14.0, start_step=1, end_step=0),
    # De-distilled transformer weights with two-branch CFG.
    "flux-de-distill": dict(steps=28, guidance=3.5, u_s=0.35, u_x=-0.45,
                            omega=7.5, start_step=1, end_step=0),
}

VIDEO_DEFAULTS = {
    "t2v": dict(steps=25, guidance=1.0, u_s=0.25, u_x=-0.25,
                omega=4.0, active_steps=7, condition_scope="joint_encoder"),
    "i2v": dict(steps=25, guidance=1.0, u_s=0.40, u_x=-0.40,
                omega=6.0, active_steps=7, condition_scope="joint_encoder"),
    # Preserve reference-image semantic tokens in the text-only variant.
    "i2v-text": dict(steps=25, guidance=1.0, u_s=0.20, u_x=-0.20,
                     omega=3.0, active_steps=7, condition_scope="text_only"),
}
