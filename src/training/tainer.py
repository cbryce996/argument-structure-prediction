import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


class Trainer:
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        valid_loader,
        test_loader,
        batch_size,
        patience=10,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.batch_size = batch_size
        self.patience = patience

    def train(self, epochs):
        best_valid_loss = float("inf")
        early_stop_counter = 0
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            total_samples = 0
            correct_train = 0
            for batch_idx, data in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, data.y)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                total_samples += len(data.y)
                _, predicted = torch.max(output, 1)
                correct_train += (predicted == data.y).sum().item()
            average_loss = total_loss / len(self.train_loader)

            self.model.eval()
            total_valid_loss = 0
            correct_valid = 0
            with torch.no_grad():
                for data in self.valid_loader:
                    output = self.model(data)
                    loss = self.criterion(output, data.y)
                    total_valid_loss += loss.item()
                    _, predicted = torch.max(output, 1)
                    correct_valid += (predicted == data.y).sum().item()
            average_valid_loss = total_valid_loss / len(self.valid_loader)

            if average_valid_loss < best_valid_loss:
                best_valid_loss = average_valid_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if early_stop_counter >= self.patience:
                break

            print(
                f"Epoch {epoch+1}/{epochs}, Training Loss: {average_loss:.4f}, Validation Loss: {average_valid_loss:.4f}"
            )

        self.model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for data in self.test_loader:
                output = self.model(data)
                predicted = torch.argmax(output, 1)
                all_labels.extend(data.y.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
        f1 = f1_score(all_labels, all_preds, average="weighted")

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted")
        recall = recall_score(all_labels, all_preds, average="weighted")

        return accuracy, precision, recall, f1
