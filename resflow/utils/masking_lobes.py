"""DEPRECATED: import from resflow.utils.masking instead.

Thin re-export shim kept for backward compatibility with existing
examples/lobes/inpainting/ code. The wells-only sampler with the
corrected z=last-axis convention now lives in masking.py.
"""
from resflow.utils.masking import (  # noqa: F401
    sample_one_well,
    generate_well_mask,
    generate_training_mask,
    apply_inpaint_output,
    InpaintDataset,
)
