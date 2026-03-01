# IMPORTS

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# DEVICE AND SEED

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# DATA INGESTION

DATA_PATH = r"https://raw.githubusercontent.com/chandanc5525/CompositeCementStrength_PredictionModel/refs/heads/main/data/raw/Concrete_Data.csv"
TARGET_COL = "Concrete compressive strength(MPa, megapascals) "

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)


# SCALING

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1,1).to(device)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1,1).to(device)


# MODEL

class ANNModel(nn.Module):

    def __init__(self, input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


model = ANNModel(X_train_scaled.shape[1]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# TRAINING WITH EPOCH TRACKING

epochs = 300
train_losses = []

for epoch in range(epochs):

    model.train()

    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    train_losses.append(loss.item())


# EPOCH GRAPH

plt.figure(figsize=(10,6))
plt.plot(train_losses)
plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("PyTorch ANN Training Curve")
plt.show()


# TEST EVALUATION

model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).cpu().numpy().ravel()

rmse = np.sqrt(mean_squared_error(y_test, preds))
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\nPyTorch ANN Performance")
print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)


# SAVE MODEL

torch.save(model.state_dict(), "models/pytorch_ann.pth")
joblib.dump(scaler, "models/scaler.pkl")