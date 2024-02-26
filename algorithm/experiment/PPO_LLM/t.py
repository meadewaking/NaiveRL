import numpy as np
import torch

import vc_models
from vc_models.models.vit import model_utils

model, embd_size, model_transforms, model_info = model_utils.load_model(model_utils.VC1_LARGE_NAME)
# To use the smaller VC-1-base model use model_utils.VC1_BASE_NAME.

# The img loaded should be Bx3x250x250
img = torch.rand([2, 3, 250, 250])

# Output will be of size Bx3x224x224
transformed_img = model_transforms(img)
# Embedding will be 1x768
embedding = model(transformed_img)
print(embedding.shape)
