import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from vc_models.models.vit import model_utils
from llarp import llama, load_safetensors, embedding


class Vision_Encoder(nn.Module):
    def __init__(self):
        super(Vision_Encoder, self).__init__()
        self.model, embd_size, self.model_transforms, model_info = model_utils.load_model(model_utils.VC1_LARGE_NAME)

    def forward(self, x):
        transformed_img = self.model_transforms(x)
        x = self.model(transformed_img)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vision_encoder = Vision_Encoder()
        self.vision_encoder.eval()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.fc_vision = nn.Linear(1024, 2048)
        self.llm_state_dict = load_safetensors(config['llm_path'],
                                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                               new_dtype=torch.float32)
        self.fc_1 = nn.Linear(2048, 256)
        self.fc_pi = nn.Linear(256, config['act_dim'])
        self.fc_v = nn.Linear(256, 1)

    def bone(self, txt, img):
        cache = {}
        img_features = []
        for i in range(img.shape[1]):
            with torch.no_grad():
                img_feature = self.vision_encoder(img[:, i, :, :, :]).view(-1, 1024)
            img_feature = F.relu(self.fc_vision(img_feature))
            img_features.append(img_feature.unsqueeze(1))
        img_features = torch.cat(img_features, dim=1)

        inputs = embedding(txt, self.llm_state_dict)
        inputs = torch.cat((inputs, img_features), dim=1)
        position_ids = torch.arange(inputs.shape[1]).view(1, -1).repeat(inputs.shape[0], 1)
        with torch.no_grad():
            out = llama(inputs, position_ids, cache, self.llm_state_dict)
        return out

    def pi(self, txt, img):
        out = self.bone(txt, img)
        x = F.relu(self.fc_1(out.view(-1, 2048)))
        x = F.layer_norm(x, normalized_shape=(256,))
        x = F.softmax(self.fc_pi(x), dim=-1)
        return x

    def v(self, txt, img):
        out = self.bone(txt, img)
        x = F.relu(self.fc_1(out.view(-1, 2048)))
        x = F.layer_norm(x, normalized_shape=(256,))
        v = self.fc_v(x)
        return v

    def pi_v(self, txt, img):
        out = self.bone(txt, img)
        x = F.relu(self.fc_1(out.view(-1, 2048)))
        x = F.layer_norm(x, normalized_shape=(256,))
        v = self.fc_v(x)
        x = F.softmax(self.fc_pi(x), dim=-1)
        return x, v
