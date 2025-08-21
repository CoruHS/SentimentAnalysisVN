import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

#Repro setup
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

#Dataset & Collate
class SentimentDataset(Dataset):
    def __init__(self, seqs, labels=None):
        self.seqs   = seqs
        self.labels = labels
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        x = torch.tensor(self.seqs[idx], dtype=torch.long)
        if self.labels is not None:
            return x, self.labels[idx]
        return x

def collate_fn(batch):
    if isinstance(batch[0], tuple):
        seqs, labels = zip(*batch)
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        seqs, labels = batch, None
    padded = pad_sequence(seqs, batch_first=True, padding_value=0)
    return (padded, labels) if labels is not None else padded

#Models 
class TextCNN(nn.Module):
    """Larger TextCNN with more kernels for stronger class boundaries"""
    def __init__(self, vocab_size, emb_dim=128, num_filters=300,
                 kernel_sizes=(2,3,4,5), dropout=0.4, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(emb_dim, num_filters, k) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)
    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        pooled = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(x))

class BiLSTM(nn.Module):
    """Bidirectional LSTM with attention‑like mean+max pooling"""
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

# helper to choose architecture
MODEL_CHOICES = {
    'cnn': TextCNN,
    'lstm': BiLSTM,
}
import torch.nn.utils as utils_nn

def train_epoch(model, loader, optimizer, criterion, scheduler, device, clip=1.0):
    model.train(); running_loss = 0.0
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward(); utils_nn.clip_grad_norm_(model.parameters(), clip)
        optimizer.step(); scheduler.step()
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

#Main

def main():
    #Load Data
    data_dir = os.path.expanduser('~/Desktop/VietnamSentiment/data')
    X      = np.load(os.path.join(data_dir, 'train_X.npy'), allow_pickle=True)
    y      = np.load(os.path.join(data_dir, 'train_Y.npy'))
    X_pub  = np.load(os.path.join(data_dir, 'public_X.npy'), allow_pickle=True)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.1,
                                                stratify=y, random_state=SEED)

    #Sampling & weights (focus on 0/2)
    counts = np.bincount(y_tr)
    class_weights = counts.mean() / counts
    class_weights[1] = 1.0  # cap neutral
    sample_w = class_weights[y_tr] ** 0.05  # almost natural freq for stable 0/2 precision
    sampler   = WeightedRandomSampler(sample_w, len(sample_w), replacement=True)

    #DataLoaders
    train_loader = DataLoader(SentimentDataset(X_tr, y_tr), batch_size=64,
                              sampler=sampler, collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(SentimentDataset(X_val, y_val), batch_size=64,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    #Model & Optim
    vocab_size = int(max(max(np.concatenate([X_tr, X_val, X_pub]))) + 1)
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model      = TextCNN(vocab_size).to(device)

    optimizer  = optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    scheduler  = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=6e-4,
                                               pct_start=0.3,
                                               steps_per_epoch=len(train_loader), epochs=25)

    crit_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    criterion    = nn.CrossEntropyLoss(weight=crit_weights)

    #Training loop
    best_f1, no_imp, patience = 0.0, 0, 6
    for epoch in range(1, 26):
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, scheduler, device)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02}  Loss={tr_loss:.4f}  ValAcc={val_acc:.4f}  ValF1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, no_imp = val_f1, 0
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"    ↳ best (F1={best_f1:.4f})")

    #Final evaluation
    model.load_state_dict(torch.load('best_model.pt'))
    val_acc, val_f1 = evaluate(model, val_loader, device)
    print(f"Best ValAcc = {val_acc:.4f}  |  Best ValF1 = {val_f1:.4f}")

    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for X, y in val_loader:
            y_true.append(y); y_pred.append(model(X.to(device)).argmax(1).cpu())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    print(classification_report(y_true, y_pred, digits=4))
    print(confusion_matrix(y_true, y_pred))

    #Public predictions
    pub_loader = DataLoader(SentimentDataset(X_pub), batch_size=64,
                            shuffle=False, collate_fn=collate_fn, num_workers=2)
    preds = []
    with torch.no_grad():
        for X in pub_loader:
            preds.append(model(X.to(device)).argmax(1).cpu().numpy())
    np.save('predictions.npy', np.concatenate(preds))
    print('Saved predictions.npy using best model')

if __name__ == '__main__':
    main()
