import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model


class Regressor(nn.Sequential):
    def __init__(self, input_channel, output_channel):
        super(Regressor, self).__init__()
        self.convA = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1)
        self.leakyreluA = nn.ReLU()
        self.convB = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1)
        self.leakyreluB = nn.ReLU()
        self.dropout = nn.Dropout(p=0.8)
        self.convC = nn.Conv2d(output_channel, 1, kernel_size=1, stride=1)
        self.activation = nn.Tanh()


    def forward(self, x):
        x = self.convA(x)
        x = self.leakyreluA(x)
        x = self.convB(x)
        x = self.leakyreluB(x)
        x = self.dropout(x)
        x = self.convC(x)

        return self.activation(x)

class MWR(nn.Module):
    def __init__(self, cfg):
        super(MWR, self).__init__()
        self.encoder = ptcv_get_model("bn_vgg16", pretrained=True)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.regressor = Regressor(1536, 512)

    def forward(self, phase, input_dic):
        if phase == 'extraction':
            im_f = self.encoder.features(input_dic['img'])
            im_f = self.avgpool(im_f)
            return im_f

        elif phase == 'comparison':
            roi_cat = torch.cat([input_dic['base_min_f'], input_dic['base_max_f'], input_dic['test_f']], dim=1)
            prediction = self.regressor(roi_cat)
            return prediction
        else:
            raise ValueError(f'Undefined phase ({phase}) has been given. It should be one of [train, val, extraction, comparison].')
