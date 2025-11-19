import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import torch
import joblib
import torch.nn as nn
from pathlib import Path
relative_path = Path("..")

train = pd.read_csv(relative_path / 'data/processed/result_train_dataset.csv')
test = pd.read_csv(relative_path / 'data/processed/result_test_dataset.csv')

numeric_train_data = train.select_dtypes(include=['number'])
numeric_test_data = test.select_dtypes(include=['number'])

X = numeric_train_data.drop(columns=['taste_cluster'])
y = train['taste_cluster']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.05, random_state=42, stratify=y, shuffle=True
)

scaler = joblib.load(relative_path / 'models/scaler.joblib')
X_train = scaler.transform(X_train)
X_val = scaler.transform(X_val)

X_test = numeric_test_data.loc[:,~numeric_test_data.columns.duplicated()].copy()
X_test = scaler.transform(X_test)

class TasteClassifier(nn.Module):
    '''
    MLP-классификатор молекул по запаху
    '''
    def __init__(self, input_dim, output_dim):
        super(TasteClassifier, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.25),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.25),

            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        out = self.block(x)
        return out
torch.manual_seed(42)
model = TasteClassifier(input_dim=X_train.shape[1], output_dim=len(y.unique()))
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
class_counts = y_train.value_counts().sort_index().values
weights = 1 / class_counts
weights = weights / weights.mean()
loss = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

model.load_state_dict(torch.load(relative_path / 'models/best_model.pth'))

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)


with torch.inference_mode():
    model.eval()
    val_logits = model(X_val_tensor)
    val_y_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)
print(f1_score(y_val_tensor, val_y_pred, average='macro'))

with torch.inference_mode():
    val_logits = model(X_test_tensor)
    val_y_pred = torch.softmax(val_logits, dim=1).argmax(dim=1)

preds = pd.DataFrame(val_y_pred.numpy(), columns=['taste_cluster'])
preds.to_csv(relative_path / 'solution.csv', index=False)