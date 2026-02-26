import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib
from init_script import SmartGridTransformer

df = pd.read_csv('CurrentVoltage.csv')
threshold = df['IL1'].mean() + (2 * df['IL1'].std())
df['is_overload'] = (df['IL1'] > threshold).astype(float)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[['VL1', 'IL1']].values)
joblib.dump(scaler, 'scaler.pkl')

WINDOW = 60
X_list, Y_list = [], []

for i in range(len(scaled_features) - WINDOW - 1):
    X_list.append(scaled_features[i:i+WINDOW])
    Y_list.append([df['is_overload'].iloc[i + WINDOW - 1], 
                   df['is_overload'].iloc[i + WINDOW]])

X_np = np.array(X_list, dtype=np.float32)
Y_np = np.array(Y_list, dtype=np.float32)

dataset = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(Y_np))
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SmartGridTransformer()
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Starting Memory-Safe Training...")
for epoch in range(10):
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/10 | Avg Loss: {total_loss/len(train_loader):.4f}")

dummy_input = torch.randn(1, WINDOW, 2)
torch.onnx.export(model, dummy_input, "smart_grid_forecast.onnx",
                  input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {0: 'batch_size'}})

print("Training complete: smart_grid_forecast.onnx created")