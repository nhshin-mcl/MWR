from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn as nn
import torch

def create_model(arg, model_name):
    ### Create model ###
    if model_name == 'Global_Regressor':
        print('Get Global_Regressor')
        model = Global_Regressor().cuda()

    if model_name == 'Local_Regressor':
        print('Get Local_Regressor')
        model = Local_Regressor(arg).cuda()

    return model

####################### Regressor Module ######################
class Regressor(nn.Sequential):
    def __init__(self, input_channel, output_channel):
        super(Regressor, self).__init__()
        self.convA = nn.Conv2d(input_channel, output_channel, kernel_size=1, stride=1)
        self.leakyreluA = nn.ReLU()
        self.convB = nn.Conv2d(output_channel, output_channel, kernel_size=1, stride=1)
        self.leakyreluB = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
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
##################################################################

########################## Total Model ###########################

class Global_Regressor(nn.Module):
    def __init__(self):
        super(Global_Regressor, self).__init__()
        self.encoder = ptcv_get_model("bn_vgg16", pretrained=True)
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.regressor = Regressor(1536, 512)

    def forward_siamese(self, x):
        x = self.encoder.features.stage1(x)
        x = self.encoder.features.stage2(x)
        x = self.encoder.features.stage3(x)
        x = self.encoder.features.stage4(x)
        x = self.encoder.features.stage5(x)
        x = self.avg_pool(x)

        return x

    def forward(self, phase, **kwargs):

        if phase == 'train':
            x_1_1, x_1_2, x_2 = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2']
            x_1_1 = self.forward_siamese(x_1_1)
            x_1_2 = self.forward_siamese(x_1_2)
            x_2 = self.forward_siamese(x_2)

            x = torch.cat([x_1_1, x_1_2, x_2], dim=1)

            output = self.regressor(x)

            return output

        elif phase == 'test':
            x_1_1, x_1_2, x_2 = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2']
            x = torch.cat([x_1_1, x_1_2, x_2], dim=1)

            output = self.regressor(x)

            return output

        elif phase == 'extraction':
            x = kwargs['x']
            x = self.forward_siamese(x)

            return x

class Local_Regressor(nn.Module):
    def __init__(self, arg):
        super(Local_Regressor, self).__init__()

        self.reg_num = arg.reg_num
        self.encoder = nn.ModuleList([ptcv_get_model("bn_vgg16", pretrained=True) for _ in range(self.reg_num)])
        self.avg_pool = nn.AvgPool2d(kernel_size=7)
        self.regressor = nn.ModuleList([Regressor(1536, 512) for _ in range(self.reg_num)])

    def forward_siamese(self, x, idx):
        x = self.encoder[idx].features(x)
        x = self.avg_pool(x)

        return x

    def forward(self, phase, **kwargs):

        if phase == 'train':
            x_1, x_2, x_test, idx = kwargs['x_1'], kwargs['x_2'], kwargs['x_test'], kwargs['idx']
            x_1, x_2, x_test = self.forward_siamese(x_1, idx), self.forward_siamese(x_2, idx), self.forward_siamese(x_test, idx)

            x_cat = torch.cat([x_1, x_2, x_test], dim=1)
            outputs = self.regressor[idx](x_cat)

            return outputs.squeeze()

        elif phase == 'test':
            x_1_1, x_1_2, x_2, idx = kwargs['x_1_1'], kwargs['x_1_2'], kwargs['x_2'], kwargs['idx']
            x = torch.cat([x_1_1, x_1_2, x_2], dim=1)
            outputs = self.regressor[idx](x)

            return outputs

        elif phase == 'extraction':
            x, idx = kwargs['x'], kwargs['idx']
            x = self.forward_siamese(x, idx)
            return x

##################################################################
