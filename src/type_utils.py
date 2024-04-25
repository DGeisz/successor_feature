import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import utils
from typing import Tuple


DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}

AttnHead = Tuple[int, int]
