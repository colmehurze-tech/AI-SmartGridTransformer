import torch
import torch.nn as nn

class SmartGridTransformer(nn.Module):
    def __init__(self, input_dim=2, model_dim=64, num_heads=4, num_layers=2):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Linear(model_dim, 2) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_fc(x)
        x = self.transformer(x)
        x = self.output_fc(x[:, -1, :]) 
        return self.sigmoid(x)

model = SmartGridTransformer()
print("Model initialized.")