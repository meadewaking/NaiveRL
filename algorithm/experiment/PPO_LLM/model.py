import torch
import torch.nn as nn
import torch.nn.functional as F
from config import config
from torchvision.models import resnet50
from llarp import llama, load_safetensors, embedding


class Vision_Encoder(nn.Module):
    def __init__(self):
        super(Vision_Encoder, self).__init__()
        a = resnet50(weights="IMAGENET1K_V2")
        self.bone = nn.Sequential(*list(resnet50(weights="IMAGENET1K_V2").children())[:-2])
        self.avg = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.bone(x)
        x = self.avg(x)
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.vision_encoder = Vision_Encoder()
        for param in self.vision_encoder.parameters():
            param.requires_grad = False
        self.fc_vision_1 = nn.Linear(2048, 2048)
        self.fc_vision_2 = nn.Linear(2048, 2048)
        self.llm_state_dict = load_safetensors(config['llm_path'],
                                               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                                               new_dtype=torch.float32)
        self.fc_1 = nn.Linear(2048, 512)
        self.fc_pi = nn.Linear(512, config['act_dim'])
        self.fc_v = nn.Linear(512, 1)

    def bone(self, txt, img):
        cache = {}
        img_features = []
        for i in range(img.shape[1]):
            img_feature = self.vision_encoder(img[:, i, :, :, :]).view(-1, 2048)
            img_feature = F.relu(self.fc_vision_1(img_feature))
            img_feature = self.fc_vision_2(img_feature)
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
        x = F.softmax(self.fc_pi(x), dim=-1)
        return x

    def v(self, txt, img):
        out = self.bone(txt, img)
        x = F.relu(self.fc_1(out.view(-1, 2048)))
        v = self.fc_v(x)
        return v

    def pi_v(self, txt, img):
        out = self.bone(txt, img)
        x = F.relu(self.fc_1(out.view(-1, 2048)))
        v = self.fc_v(x)
        x = F.softmax(self.fc_pi(x), dim=-1)
        return x, v
