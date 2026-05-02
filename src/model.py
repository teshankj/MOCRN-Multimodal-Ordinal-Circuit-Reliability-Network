import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import f1_score, accuracy_score
# ==============================================================================
# STEP 1: FLEXIBLE EORN MODEL ARCHITECTURE
# ==============================================================================

class FlexibleEORN_LateFusionModel(nn.Module):
    """
    Flexible EORN that auto-adapts to any circuit dimensions
    """
    def __init__(self, wave_channels=3, stat_dim=61, num_ft_classes=6,
                 num_deg_levels=6, seq_len=200, dropout_rate=0.3):
        super().__init__()

        self.wave_channels = wave_channels
        self.stat_dim = stat_dim

        # Waveform branch
        self.conv1 = nn.Conv1d(wave_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(128)

        self.adaptive_pool = nn.AdaptiveMaxPool1d(64)
        self.wave_flat_size = 128 * 64

        self.wave_fc_shared = nn.Linear(self.wave_flat_size, 256)
        self.wave_fc_ft_path = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.wave_fc_deg_path = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.wave_fc_ft = nn.Linear(128, num_ft_classes)
        self.wave_evidence_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_deg_levels)
        )

        # Statistical branch
        self.stat_fc1 = nn.Linear(stat_dim, 64)
        self.stat_bn1 = nn.BatchNorm1d(64)
        self.stat_fc2 = nn.Linear(64, 128)
        self.stat_bn2 = nn.BatchNorm1d(128)
        self.stat_fc_shared = nn.Linear(128, 256)

        self.stat_fc_ft_path = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.stat_fc_deg_path = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        self.stat_fc_ft = nn.Linear(128, num_ft_classes)
        self.stat_evidence_net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_deg_levels)
        )

        # Fusion weights
        self.fusion_weight_ft = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.fusion_weight_deg = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.evidence_scale = nn.Parameter(torch.tensor(10.0))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        for layer in [self.wave_evidence_net[-1], self.stat_evidence_net[-1]]:
            nn.init.normal_(layer.weight, mean=0, std=0.01)
            nn.init.constant_(layer.bias, 2.0)

    def forward(self, x_wave, x_stat):
        # Waveform branch
        x_w = F.relu(self.bn1(self.conv1(x_wave)))
        x_w = F.max_pool1d(x_w, 2)
        x_w = F.relu(self.bn2(self.conv2(x_w)))
        x_w = F.max_pool1d(x_w, 2)
        x_w = F.relu(self.bn3(self.conv3(x_w)))
        x_w = F.relu(self.bn4(self.conv4(x_w)))
        x_w = self.adaptive_pool(x_w)
        x_w = x_w.view(x_w.size(0), -1)
        x_w = F.relu(self.wave_fc_shared(x_w))

        x_w_ft = self.wave_fc_ft_path(x_w)
        x_w_deg = self.wave_fc_deg_path(x_w)
        out_ft_w = self.wave_fc_ft(x_w_ft)
        evidence_w = self.wave_evidence_net(x_w_deg)

        # Statistical branch
        x_s = F.relu(self.stat_bn1(self.stat_fc1(x_stat)))
        x_s = F.relu(self.stat_bn2(self.stat_fc2(x_s)))
        x_s = F.relu(self.stat_fc_shared(x_s))

        x_s_ft = self.stat_fc_ft_path(x_s)
        x_s_deg = self.stat_fc_deg_path(x_s)
        out_ft_s = self.stat_fc_ft(x_s_ft)
        evidence_s = self.stat_evidence_net(x_s_deg)

        # Late fusion
        w_ft = F.softmax(self.fusion_weight_ft, dim=0)
        w_deg = F.softmax(self.fusion_weight_deg, dim=0)

        out_ft = w_ft[0] * out_ft_w + w_ft[1] * out_ft_s
        evidence = w_deg[0] * evidence_w + w_deg[1] * evidence_s

        evidence = F.softplus(evidence) * torch.abs(self.evidence_scale)
        alpha = evidence + 1.0

        return out_ft, alpha