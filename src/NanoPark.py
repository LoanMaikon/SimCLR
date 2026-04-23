import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, groups=1, act_layer=nn.ReLU):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer(inplace=True) if act_layer is nn.ReLU else act_layer(),
        )

    def forward(self, x):
        return self.block(x)

class SqueezeExcite(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        hidden = max(8, ch // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(ch, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, ch, 1)

    def forward(self, x):
        s = self.pool(x)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class UniversalInvertedResidual(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        stride=1,
        expand_ratio=4,
        kernel_size=3,
        use_se=False,
        act_layer=nn.ReLU,
    ):
        super().__init__()

        hidden_ch = in_ch * expand_ratio
        self.use_residual = (stride == 1 and in_ch == out_ch)

        layers = []

        # Expansion
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_ch, hidden_ch, kernel_size=1, stride=1, act_layer=act_layer))

        # Depthwise
        layers.append(
            ConvBNAct(
                hidden_ch,
                hidden_ch,
                kernel_size=kernel_size,
                stride=stride,
                groups=hidden_ch,
                act_layer=act_layer,
            )
        )

        # Squeeze-and-Excitation
        layers.append(SqueezeExcite(hidden_ch) if use_se else nn.Identity())

        # Projection
        layers.append(nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = x
        
        out = self.block(out)

        if self.use_residual:
            out = out + x

        return out

class NanoPark(nn.Module):
    def __init__(self, in_channels=3, width_multiplier=1.0):
        super().__init__()

        self.width_multiplier = width_multiplier

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, int(16*width_multiplier), kernel_size=3, stride=2, act_layer=nn.ReLU),
        )
        self.uib1 = nn.Sequential(
            UniversalInvertedResidual(int(16*width_multiplier), int(16*width_multiplier), stride=1, expand_ratio=2, kernel_size=3, use_se=True),
        )

        self.uib2 = nn.Sequential(
            UniversalInvertedResidual(int(16*width_multiplier), int(24*width_multiplier), stride=2, expand_ratio=2, kernel_size=3, use_se=True),
            UniversalInvertedResidual(int(24*width_multiplier), int(24*width_multiplier), stride=1, expand_ratio=2, kernel_size=3, use_se=True),
        )

        self.uib3 = nn.Sequential(
            UniversalInvertedResidual(int(24*width_multiplier), int(32*width_multiplier), stride=2, expand_ratio=4, kernel_size=5, use_se=False),
            UniversalInvertedResidual(int(32*width_multiplier), int(32*width_multiplier), stride=1, expand_ratio=4, kernel_size=5, use_se=False),
        )

        self.uib4 = nn.Sequential(
            UniversalInvertedResidual(int(32*width_multiplier), int(48*width_multiplier), stride=2, expand_ratio=4, kernel_size=3, use_se=False),
            UniversalInvertedResidual(int(48*width_multiplier), int(48*width_multiplier), stride=1, expand_ratio=4, kernel_size=3, use_se=False),
        )

        self.head = nn.Sequential(
            ConvBNAct(int(48*width_multiplier), int(64*width_multiplier), kernel_size=1, stride=1, act_layer=nn.ReLU),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Identity() # Placeholder for potential future fully connected layer

        self._load_projection_head()

    def forward(self, x):
        x = self.stem(x)
        x = self.uib1(x)
        x = self.uib2(x)
        x = self.uib3(x)
        x = self.uib4(x)
        x = self.head(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
    def fit_classifier_head(self, num_classes):
        self.fc = nn.Linear(int(64*self.width_multiplier), num_classes, bias=True)

    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)

        errors = []

        try:
            self.load_state_dict(state_dict)
            return
        except Exception as e:
            errors.append(("direct_load", str(e)))

        # fc = 2
        self.fit_classifier_head(num_classes=2)
        try:
            self.load_state_dict(state_dict)
            return
        except Exception as e:
            errors.append(("classifier_head_2", str(e)))

        # fc = 1000
        self.fit_classifier_head(num_classes=1000)
        try:
            self.load_state_dict(state_dict)
            return
        except Exception as e:
            errors.append(("classifier_head_1000", str(e)))

        raise RuntimeError(f"Failed to load weights from {weight_path} with the following errors: {errors}")
    
    def freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def unfreeze_layers(self, unfrozen_layers):
        match unfrozen_layers:
            case "all":
                for param in self.parameters():
                    param.requires_grad = True
         
            case _:
                raise ValueError(f"Unfrozen layers option {self.unfrozen_layers} not recognized.")
    
    def fit_projection_head(self):
        self.fc = self.projection_head

    '''
    Following https://github.com/google-research/simclr/blob/master/model_util.py
    '''
    def _load_projection_head(self):
        match self.projection_head_mode:
            case 'linear':
                self.projection_head = nn.Sequential(
                    nn.Linear(self.encoder_out_features, self.projection_dim),
                    nn.BatchNorm1d(self.projection_dim),
                )


            case 'non-linear':
                self.projection_head = nn.Sequential(
                    nn.Linear(self.encoder_out_features, self.encoder_out_features),
                    nn.BatchNorm1d(self.encoder_out_features),
                    nn.ReLU(),
                    nn.Linear(self.encoder_out_features, self.projection_dim),
                    nn.BatchNorm1d(self.projection_dim),
                )
            
            case 'none':
                self.projection_head = nn.Identity()
        
        if self.projection_head is None:
            raise ValueError("Projection head mode must be 'linear', 'non-linear', or 'none'.")

def nanopark_large(in_channels=3):
    return NanoPark(in_channels=in_channels, width_multiplier=1.5)

def nanopark_base(in_channels=3):
    return NanoPark(in_channels=in_channels, width_multiplier=1.0)

def nanopark_small(in_channels=3):
    return NanoPark(in_channels=in_channels, width_multiplier=0.5)

def nanopark_tiny(in_channels=3):
    return NanoPark(in_channels=in_channels, width_multiplier=0.25)
