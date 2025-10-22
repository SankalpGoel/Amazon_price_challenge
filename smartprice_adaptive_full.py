
import os
import argparse
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# ---------------- LOGGING SETUP ----------------
logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

# ---------------- METRIC FUNCTIONS ----------------
def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def print_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape_val = smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logging.info(f"📊 MAE={mae:.3f} | RMSE={rmse:.3f} | SMAPE={smape_val:.2f}% | R²={r2:.3f}")

# ---------------- DATASET ----------------
class PriceDataset(Dataset):
    def __init__(self, texts, targets=None, tokenizer=None, max_len=128):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.targets is not None:
            item["labels"] = torch.tensor(self.targets[idx], dtype=torch.float)
        return item

# ---------------- MODEL ----------------
class PriceModel(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = outputs.last_hidden_state.mean(dim=1)
        return self.regressor(emb).squeeze(-1)

# ---------------- TRAINING FUNCTION ----------------
def train_model(train_df, folds=5, lr=2e-5, adaptive=True, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    logging.info("🚀 Starting Training with Cross-Validation...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    fold_preds, fold_metrics = [], []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
        logging.info(f"\n===== FOLD {fold + 1}/{folds} =====")
        train_texts = train_df.iloc[tr_idx]["catalog_content"].tolist()
        val_texts = train_df.iloc[val_idx]["catalog_content"].tolist()
        train_targets = train_df.iloc[tr_idx]["price"].values
        val_targets = train_df.iloc[val_idx]["price"].values

        model = PriceModel(model_name)
        model.cuda()
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.L1Loss()

        train_dataset = PriceDataset(train_texts, train_targets, tokenizer)
        val_dataset = PriceDataset(val_texts, val_targets, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

        # Adaptive learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=2, verbose=True)

        best_val = float('inf')
        for epoch in range(3):  # keep light for demo
            model.train()
            running_loss = 0.0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                optimizer.zero_grad()
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(batch["input_ids"], batch["attention_mask"])
                loss = criterion(outputs, batch["labels"])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            logging.info(f"Epoch {epoch+1} | Avg Train Loss: {running_loss / len(train_loader):.4f}")

            # Validation
            model.eval()
            preds, targets = [], []
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.cuda() for k, v in batch.items()}
                    outputs = model(batch["input_ids"], batch["attention_mask"])
                    preds.extend(outputs.cpu().numpy())
                    targets.extend(batch["labels"].cpu().numpy())

            val_loss = mean_absolute_error(targets, preds)
            scheduler.step(val_loss)

            logging.info(f"Validation Loss: {val_loss:.4f}")
            print_metrics(np.array(targets), np.array(preds))

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), f"model_fold{fold}.bin")

        fold_metrics.append(best_val)

    logging.info(f"\n✅ Training Complete! Avg Val MAE: {np.mean(fold_metrics):.4f}")
    return model_name

# ---------------- PREDICTION FUNCTION ----------------
def predict(test_df, model_name, folds=5):
    logging.info("🧠 Generating predictions on test set...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    test_dataset = PriceDataset(test_df["catalog_content"].tolist(), tokenizer=tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    preds_all = []
    for fold in range(folds):
        model = PriceModel(model_name)
        if os.path.exists(f"model_fold{fold}.bin"):
            model.load_state_dict(torch.load(f"model_fold{fold}.bin"))
        model.cuda()
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Inference Fold {fold+1}"):
                batch = {k: v.cuda() for k, v in batch.items()}
                outputs = model(batch["input_ids"], batch["attention_mask"])
                preds.extend(outputs.cpu().numpy())

        preds_all.append(preds)

    final_preds = np.mean(preds_all, axis=0)
    test_df["price"] = np.maximum(0, final_preds)  # ensure positive

    out_path = "dataset/test_out.csv"
    test_df[["sample_id", "price"]].to_csv(out_path, index=False)
    logging.info(f"✅ Predictions saved to {out_path}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--images-dir", type=str, default="dataset/images")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=2048)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    train_path = "dataset/train.csv"
    test_path = "dataset/test.csv"
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if args.train:
        model_name = train_model(train_df, folds=args.folds)
    if args.predict:
        predict(test_df, model_name="sentence-transformers/all-MiniLM-L6-v2", folds=args.folds)
