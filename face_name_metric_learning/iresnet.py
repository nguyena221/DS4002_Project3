"""PyTorch IResNet-100 backbone used by the metric-learning notebooks."""


def build_iresnet100():
    """Build an InsightFace-compatible 512-dimensional IResNet-100."""
    try:
        import torch
        from torch import nn
    except ImportError as error:
        raise ImportError("candidate training requires PyTorch") from error

    class IBasicBlock(nn.Module):
        expansion = 1

        def __init__(self, in_planes, planes, stride=1):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes, eps=1e-5)
            self.conv1 = nn.Conv2d(
                in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
            self.prelu = nn.PReLU(planes)
            self.conv2 = nn.Conv2d(
                planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
            )
            self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
            self.downsample = None
            if stride != 1 or in_planes != planes:
                self.downsample = nn.Sequential(
                    nn.Conv2d(
                        in_planes,
                        planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes, eps=1e-5),
                )

        def forward(self, value):
            identity = (
                self.downsample(value) if self.downsample is not None else value
            )
            out = self.bn1(value)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.prelu(out)
            out = self.conv2(out)
            out = self.bn3(out)
            return out + identity

    class IResNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_planes = 64
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
            self.prelu = nn.PReLU(64)
            self.layer1 = self._make_layer(64, 3, 2)
            self.layer2 = self._make_layer(128, 13, 2)
            self.layer3 = self._make_layer(256, 30, 2)
            self.layer4 = self._make_layer(512, 3, 2)
            self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
            self.dropout = nn.Dropout2d(0.4)
            self.fc = nn.Linear(512 * 7 * 7, 512)
            self.features = nn.BatchNorm1d(512, eps=1e-5)

        def _make_layer(self, planes, blocks, stride):
            layers = [IBasicBlock(self.in_planes, planes, stride)]
            self.in_planes = planes
            layers.extend(IBasicBlock(self.in_planes, planes) for _ in range(1, blocks))
            return nn.Sequential(*layers)

        def forward(self, value):
            value = self.conv1(value)
            value = self.bn1(value)
            value = self.prelu(value)
            value = self.layer1(value)
            value = self.layer2(value)
            value = self.layer3(value)
            value = self.layer4(value)
            value = self.bn2(value)
            value = self.dropout(value)
            value = torch.flatten(value, 1)
            value = self.fc(value)
            return self.features(value)

    return IResNet()


def load_iresnet100(checkpoint_path, device):
    """Load pretrained IResNet-100 weights without downloading anything."""
    import torch

    model = build_iresnet100()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = (
        checkpoint.get("backbone_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint
    )
    state = {
        key.removeprefix("module."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return model
