import torch.nn as nn

class SFAM(nn.Module):
    def __init__(self, planes, num_levels, num_scales, compress_ratio=16):
        super(SFAM, self).__init__()
        self.planes = planes
        self.num_levels = num_levels
        self.num_scales = num_scales
        self.compress_ratio = compress_ratio

        self.fc1 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels,
                                                 self.planes*self.num_levels // self.compress_ratio,
                                                 1, 1, 0)] * self.num_scales)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.ModuleList([nn.Conv2d(self.planes*self.num_levels // self.compress_ratio,
                                                 self.planes*self.num_levels,
                                                 1, 1, 0)] * self.num_scales)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        attention_feat = []
        for i, _mf in enumerate(x):
            _tmp_f = self.avgpool(_mf)
            _tmp_f = self.fc1[i](_tmp_f)
            _tmp_f = self.relu(_tmp_f)
            _tmp_f = self.fc2[i](_tmp_f)
            _tmp_f = self.sigmoid(_tmp_f)
            attention_feat.append(_mf*_tmp_f)
        return attention_feat