import torch
from sklearn.metrics import accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training function (epoch-wise printing)
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, patience=3):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_acc = 0
    epochs_no_improve = 0
    train_acc_list, val_acc_list = [], []

    for epoch in range(1, epochs+1):
        model.train()
        train_preds, train_labels = [], []

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].float().to(device)

            optimizer.zero_grad()
            outputs = model(input_ids)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = (outputs > 0.5).long()
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())

        # Epoch metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_acc_list.append(train_acc)

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["label"].float().to(device)
                outputs = model(input_ids)
                preds = (outputs > 0.5).long()
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_acc = accuracy_score(val_labels, val_preds)
        val_acc_list.append(val_acc)

        print(f"Epoch {epoch}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    return train_acc_list, val_acc_list
