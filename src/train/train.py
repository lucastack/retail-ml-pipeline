import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from google.cloud import storage


parser = argparse.ArgumentParser()
parser.add_argument(
    "--output_path", type=str, required=True, help="output path for the model."
)
parser.add_argument(
    "--dataset_path",
    type=str,
    required=True,
    help="dataset path for training.",
)
args = parser.parse_args()


df = pd.read_csv(args.dataset_path)
embeddings_model = SentenceTransformer(
    "sentence-transformers/paraphrase-albert-small-v2"
)
text_embeddings = embeddings_model.encode(df["description"])
cat_features = df[["brand_name"]]
encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
cat_encoded = encoder.fit_transform(cat_features)
X = np.hstack([text_embeddings, cat_encoded])


def standardize(data):
    mean = data.mean()
    std = data.std()
    return (data - mean) / std, mean, std


y = df["price"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train_standardized, mean_y, std_y = standardize(y_train)
y_test_standardized = (y_test - mean_y) / std_y

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train_standardized, dtype=torch.float32).view(-1, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test_standardized, dtype=torch.float32).view(-1, 1)


class LingerieDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


train_dataset = LingerieDataset(X_train_t, y_train_t)
test_dataset = LingerieDataset(X_test_t, y_test_t)


class LinguerieModel(nn.Module):
    def __init__(self, input_dim, num_layers=5, hidden_dim=128):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


model = LinguerieModel(input_dim=775, hidden_dim=128)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
epochs = 100
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
model.train()


def mean_absolute_percentage_error(y_true, y_pred):
    percentage_errors = torch.abs((y_true - y_pred) / y_true) * 100
    return torch.mean(percentage_errors).item()


for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} complete.")

    total_mape = 0
    count_batches = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            batch_y_reescaled = std_y * batch_y + mean_y
            preds_reescaled = std_y * preds + mean_y
            percentage_errors = (
                torch.abs(
                    (batch_y_reescaled - preds_reescaled) / batch_y_reescaled
                )
                * 100
            )
            mape = torch.mean(percentage_errors).item()
            total_mape += mape
            count_batches += 1
    average_mape = total_mape / count_batches
    print(average_mape)


torch.save(model.state_dict(), "model.pt")


def upload_to_gcs(local_path, gcs_uri):
    storage_client = storage.Client()
    bucket_name, blob_path = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_filename(local_path)


upload_to_gcs("model.pt", args.output_path)
