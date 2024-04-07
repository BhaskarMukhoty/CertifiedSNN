from sklearn.semi_supervised import SelfTrainingClassifier
from models.layers import *
import torchvision

cfg = {
    'vgg5' : [[64, 'A'], 
              [128, 128, 'A'],
              [],
              [],
              []],
    'vgg11': [
        [64, 'A'],
        [128, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512],
        []
    ],
    'vgg13': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 'A'],
        [512, 512, 'A'],
        [512, 512, 'A']
    ],
    'vgg16': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 'A'],
        [512, 512, 512, 'A'],
        [512, 512, 512, 'A']
    ],
    'vgg19': [
        [64, 64, 'A'],
        [128, 128, 'A'],
        [256, 256, 256, 256, 'A'],
        [512, 512, 512, 512, 'A'],
        [512, 512, 512, 512, 'A']
    ]
}

class VGG(nn.Module):
    def __init__(self, vgg_name, encoding, atk_encoding, model_mode, T, num_class, norm, init_c=3):
        super(VGG, self).__init__()
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.T = T
        self.init_channels = init_c
        self.encoding = encoding
        self.atk_encoding = atk_encoding
        self.model_mode=model_mode

        if vgg_name == 'vgg11' or vgg_name == 'vgg5':
            self.W = 16 
        else:
            self.W = 1
        
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])
        self.classifier = self._make_classifier(num_class)
        
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(x))
                layers.append(LIFSpike(self.T))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [nn.Flatten(), nn.Linear(512*self.W, 4096), LIFSpike(self.T), nn.Linear(4096, 4096), LIFSpike(self.T), nn.Linear(4096, num_class)]    
        return nn.Sequential(*layer)
    
    #pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return
    
    def get_encoding(self, encoding, input):
        #print(encoding)
        if encoding == 'rate':
            input = rate_encode(input, self.T)      #rate encoding
            # input = rate_encode(input,  self.T,labels =None,atk=None,model=None,eps=None)
        elif encoding == 'const':
            input = self.norm(input)
            input = const_encode(input, self.T)    #constant encoding
        else:
            print("--encoding/atk_encoding not reconginzed")
        return input

    def forward(self, input):
        #print(self.T)    
        if self.T > 0:
            
            #input = add_dimention(input, self.T)
            #input = self.merge(input)    
            #print(self.model_mode)
            if self.model_mode=='normal':
                    input = self.get_encoding(self.encoding, input)
                    # input = rate_encode(input,  self.T,labels,atk=True,model=model2)
            elif self.model_mode=='attack':
                    input = self.get_encoding(self.atk_encoding, input)
            else:
                print("--mode not specified. normal/attack")
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        # print(out.shape)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out
    

class VGG_woBN(nn.Module):
    def __init__(self, vgg_name, T, num_class, norm, dropout=0.1, init_c=3):
        super(VGG_woBN, self).__init__()
        if norm is not None and isinstance(norm, tuple):
            self.norm = TensorNormalization(*norm)
        else:
            self.norm = TensorNormalization((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.T = T
        self.init_channels = init_c
        self.dropout = dropout

        if "wobn" in vgg_name:
            vgg_name = 'vgg5'
        if vgg_name == 'vgg11' or vgg_name == 'vgg5':
            self.W = 16
        else:
            self.W = 1
        
        self.layer1 = self._make_layers(cfg[vgg_name][0])
        self.layer2 = self._make_layers(cfg[vgg_name][1])
        self.layer3 = self._make_layers(cfg[vgg_name][2])
        self.layer4 = self._make_layers(cfg[vgg_name][3])
        self.layer5 = self._make_layers(cfg[vgg_name][4])
        self.classifier = self._make_classifier(num_class)
        
        self.merge = MergeTemporalDim(T)
        self.expand = ExpandTemporalDim(T)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg):
        layers = []
        for x in cfg:
            if x == 'A':
                layers.append(nn.AvgPool2d(2))
            else:
                layers.append(nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1))
                layers.append(LIFSpike(self.T))
                layers.append(nn.Dropout(self.dropout))
                self.init_channels = x
        return nn.Sequential(*layers)

    def _make_classifier(self, num_class):
        layer = [nn.Flatten(), nn.Linear(512*self.W, 4096), LIFSpike(self.T), nn.Linear(4096, 4096), LIFSpike(self.T), nn.Linear(4096, num_class)]    
        return nn.Sequential(*layer)
    
    # pass T to determine whether it is an ANN or SNN
    def set_simulation_time(self, T, mode='bptt'):
        self.T = T
        for module in self.modules():
            if isinstance(module, (LIFSpike, ExpandTemporalDim)):
                module.T = T
                if isinstance(module, LIFSpike):
                    module.mode = mode
        return

    def forward(self, input):
        input = self.norm(input)
        if self.T > 0:
            input = add_dimention(input, self.T)
            input = self.merge(input)
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.classifier(out)
        if self.T > 0:
            out = self.expand(out)
        return out

if __name__ == '__main__':
    model = VGG('vgg11', 'rate', 'rate', 'normal', 4, 10, None, init_c=3)
    x = torch.rand(64,3,32,32)
    labels = torch.rand(64)
    print(model(x,labels))