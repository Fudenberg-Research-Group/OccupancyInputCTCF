import torch
import torch.nn as nn
torch.manual_seed(2024)

class FlankCoreModel(nn.Module):
    def __init__(self, seq_len, n_head, kernel_size, n_feature=4):
        super().__init__()

        d_embed1 = 5
        d_embed2 = 5
        maxpool_size = 5
        d_embed3 = 11

        # capture core
        self.convblock1 = nn.Sequential(
            nn.Conv1d(in_channels=n_feature, out_channels=d_embed1, kernel_size=18, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=d_embed1),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )

        # capture flank
        self.convblock2 = nn.Sequential(
            nn.Conv1d(in_channels=n_feature, out_channels=d_embed2, kernel_size=15, padding='same'),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=d_embed2),
            nn.MaxPool1d(kernel_size=maxpool_size, stride=maxpool_size)
        )

        # convolve on the new features
        self.convblock3 = nn.Sequential(
            nn.Conv1d(in_channels=d_embed1+d_embed2, out_channels=d_embed3, kernel_size=5, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=d_embed3),
        )

        self.flatten = nn.Flatten()
        flatten_out_features = self._calculate_num_out_features(n_feature, seq_len)
        self.linear = nn.Linear(in_features=flatten_out_features, out_features=2)

    def _calculate_num_out_features(self, n_feature, seq_len):
        with torch.no_grad():  
            x = torch.zeros(1, n_feature, seq_len)
            
            out_1 = self.convblock1(x)
            out_2 = self.convblock2(x)
            out = torch.cat([out_1, out_2], dim=1)
            out = self.convblock3(out)
            out = self.flatten(out)

            out_features = out.shape[1]

            return out_features
        
    def forward(self, x):
        out_1 = self.convblock1(x)
        out_2 = self.convblock2(x)
        out = torch.cat([out_1, out_2], dim=1)
        out = self.convblock3(out)
        out = self.flatten(out)
        out = self.linear(out)
        
        return out