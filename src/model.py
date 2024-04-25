from transformer_lens import (
    HookedTransformer,
)

from src.type_utils import DTYPES
from src.encoder import encoder

model = (
    HookedTransformer.from_pretrained(encoder.cfg["model_name"])
    .to(DTYPES[encoder.cfg["enc_dtype"]])
    .to(encoder.cfg["device"])
)
