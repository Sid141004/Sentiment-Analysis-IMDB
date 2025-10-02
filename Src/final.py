# Step 3: Define multiple models
model_relu_dropout = FFNN(vocab_size=tokenizer.vocab_size, activation="relu", regularization="dropout").to(device)
model_relu_nodrop = FFNN(vocab_size=tokenizer.vocab_size, activation="relu", regularization=None).to(device)
model_tanh_dropout = FFNN(vocab_size=tokenizer.vocab_size, activation="tanh", regularization="dropout").to(device)

train_acc_relu, val_acc_relu = train_model(model_relu_dropout, train_loader, test_loader, epochs=10, lr=1e-3, patience=3)
train_acc_relu_no, val_acc_relu_no = train_model(model_relu_nodrop, train_loader, test_loader, epochs=10, lr=1e-3, patience=3)
train_acc_tanh_drop, val_acc_tanh_drop = train_model(model_tanh_dropout, train_loader, test_loader, epochs=10, lr=1e-3, patience=3)

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

plt.figure(figsize=(12,6))

# Training vs Validation curves
plt.plot(train_acc_relu, linestyle='--', label="ReLU+Dropout Train")
plt.plot(val_acc_relu, label="ReLU+Dropout Val")

plt.plot(train_acc_relu_no, linestyle='--', label="ReLU+NoDrop Train")
plt.plot(val_acc_relu_no, label="ReLU+NoDrop Val")

plt.plot(train_acc_tanh_drop, linestyle='--', label="Tanh+Dropout Train")
plt.plot(val_acc_tanh_drop, label="Tanh+Dropout Val")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy - Activation/Regularization Comparison")
plt.legend()
plt.grid(True)
plt.show()

def get_classification_report(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].cpu().numpy()
            outputs = model(input_ids)
            preds = (outputs > 0.5).long().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return classification_report(all_labels, all_preds, target_names=["Negative","Positive"])

print("=== ReLU + Dropout ===")
print(get_classification_report(model_relu_dropout, test_loader))

print("\n=== ReLU + No Dropout ===")
print(get_classification_report(model_relu_nodrop, test_loader))

print("\n=== Tanh + Dropout ===")
print(get_classification_report(model_tanh_dropout, test_loader))
