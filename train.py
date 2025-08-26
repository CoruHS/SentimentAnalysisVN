import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import torch.nn.utils as utils_nn
from itertools import chain

# Repro
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#Models
class SentimentDataset(Dataset):
    def __init__(self, seqs, labels=None):
        self.seqs   = seqs
        self.labels = labels
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        if self.labels is not None:
            return x, int(self.labels[idx])
        return x

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        seqs, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        seqs, labels = batch, None
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return (padded, labels) if labels is not None else padded

class TextCNN(nn.Module):
    """Larger TextCNN with more kernels for stronger class boundaries"""
    def __init__(self, vocab_size, emb_dim=128, num_filters=300,
                 kernel_sizes=(2,3,4,5), dropout=0.4, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        pooled = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(x))

class BiLSTM(nn.Module):
    """Bidirectional LSTM with attention-like mean+max pooling"""
    def __init__(self, vocab_size, emb_dim=128, hidden=256, num_layers=1, dropout=0.4, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden*2*2, num_classes)  # mean+max pool concat
    def forward(self, x):
        emb = self.embedding(x)
        out, _ = self.lstm(emb)
        mean_pool = out.mean(dim=1)
        max_pool  = out.max(dim=1)[0]
        cat = torch.cat([mean_pool, max_pool], dim=1)
        return self.fc(self.dropout(cat))

MODEL_CHOICES = {
    'cnn': TextCNN,
    'lstm': BiLSTM,
}

def train_epoch(model, loader, optimizer, criterion, scheduler, device, clip=1.0):
    model.train(); running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        utils_nn.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running_loss += loss.item() * X.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval(); preds, trues = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            preds.append(model(X).argmax(1).cpu())
            trues.append(y)
    y_pred = torch.cat(preds).numpy(); y_true = torch.cat(trues).numpy()
    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average='macro')

# small utility to get robust vocab size for ragged seq
def max_token_over(*splits):
    m = 0
    for seq in chain.from_iterable(splits):
        if len(seq):
            mv = int(max(seq))
            if mv > m: m = mv
    return m

# Main
def main():
    data_dir = os.path.expanduser('~/Desktop/VietnamSentiment/data')

    # Load (object arrays of lists)
    X      = np.load(os.path.join(data_dir, 'train_X.npy'), allow_pickle=True)
    y      = np.load(os.path.join(data_dir, 'train_Y.npy'))
    X_pub  = np.load(os.path.join(data_dir, 'test_X.npy'),  allow_pickle=True)

    # ==== FIX 1: remap labels to contiguous 0..K-1 and record mapping ====
    classes = np.sort(np.unique(y))
    class_to_idx = {int(c): i for i, c in enumerate(classes)}
    y = np.array([class_to_idx[int(c)] for c in y], dtype=np.int64)
    num_classes = len(classes)
    print(f"[INFO] Original labels: {classes} -> remapped to 0..{num_classes-1}")

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=SEED
    )

    # ==== FIX 2: make class_weights match num_classes exactly ====
    counts = np.bincount(y_tr, minlength=num_classes)
    safe_counts = np.maximum(counts, 1)                  # avoid div-by-zero
    class_weights = safe_counts.mean() / safe_counts      # inverse freq-ish

    # If your original dataset used label "1" for neutral, cap its weight at 1.0
    if 1 in class_to_idx:
        neutral_idx = class_to_idx[1]
        class_weights[neutral_idx] = 1.0

    # Smooth the sampler weights a bit to avoid extreme skew
    sample_w = (class_weights[y_tr]) ** 0.05
    sampler  = WeightedRandomSampler(
        weights=torch.as_tensor(sample_w, dtype=torch.double),
        num_samples=len(sample_w),
        replacement=True
    )

    # Dataloaders
    train_loader = DataLoader(SentimentDataset(X_tr, y_tr), batch_size=64,
                              sampler=sampler, collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(SentimentDataset(X_val, y_val), batch_size=64,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Model & Optim
    vocab_size = max_token_over(X_tr, X_val, X_pub) + 1   # ==== FIX 3 ====
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model      = TextCNN(vocab_size, num_classes=num_classes).to(device)  # ==== FIX 4 ====

    optimizer  = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=6e-4, pct_start=0.3,
        steps_per_epoch=len(train_loader), epochs=25
    )

    crit_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    criterion    = nn.CrossEntropyLoss(weight=crit_weights)  # matches num_classes

    # Training
    best_f1, no_imp = 0.0, 0
    for epoch in range(1, 26):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02}  Loss={tr_loss:.4f}  ValAcc={val_acc:.4f}  ValF1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, no_imp = val_f1, 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"    ↳ best (F1={best_f1:.4f})")
        else:
            no_imp += 1

    # Final eval
    model.load_state_dict(torch.load('best_model.pt', map_location=device))
    val_acc, val_f1 = evaluate(model, val_loader, device)
    print(f"Best ValAcc = {val_acc:.4f}  |  Best ValF1 = {val_f1:.4f}")

    # Detailed report
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in val_loader:
            y_true.append(yb.numpy())
            y_pred.append(model(Xb.to(device)).argmax(1).cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))

    # Public predictions
    pub_loader = DataLoader(SentimentDataset(X_pub), batch_size=64,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    preds = []
    with torch.no_grad():
        for Xb in pub_loader:
            preds.append(model(Xb.to(device)).argmax(1).cpu().numpy())
    np.save('predictions.npy', np.concatenate(preds))
    print('Saved predictions.npy using best model')

if __name__ == '__main__':
    main()
