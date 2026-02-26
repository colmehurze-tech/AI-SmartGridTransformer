import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class SmartGridDataset(Dataset):
    def __init__(self, file_path, window_size=60):
        df = pd.read_csv(file_path)
        self.data = df[['VL1', 'IL1']].values 
        self.data = (self.data - self.data.mean(axis=0)) / self.data.std(axis=0)        
        self.window_size = window_size

    def __len__(self):
        return len(self.data) - self.window_size

    def __getitem__(self, idx):
        window = self.data[idx : idx + self.window_size]
        future_current = self.data[idx + self.window_size][1]
        target = 1.0 if future_current > 2.0 else 0.0
        return torch.tensor(window, dtype=torch.float32), torch.tensor([target], dtype=torch.float32)

file_path = "CurrentVoltage.csv" 
dataset = SmartGridDataset(file_path)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

print(f"The Dataset has been loaded. Total sequences: {len(dataset)}")