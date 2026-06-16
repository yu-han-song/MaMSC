import os
import torch
import torch.nn as nn
from torch import optim
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from sklearn.metrics import accuracy_score
from exp.exp_basic import Exp_Basic

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super(Exp_Classification, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_gpu and self.args.use_multi_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss()

    def train(self, setting):
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        train_data, train_loader = self._get_data('train')
        val_data, val_loader = self._get_data('val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for batch in train_loader:

                batch_x, batch_y = batch
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)

                model_optim.zero_grad()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                model_optim.step()

                epoch_loss += loss.item()
                pred = outputs.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

            epoch_loss /= len(train_loader)
            acc = correct / total
            val_acc, val_loss = self.evaluate(val_loader, criterion)

            print(
                f"Epoch {epoch + 1} | Train Loss: {epoch_loss:.4f} | Acc: {acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

            train_losses.append(epoch_loss)
            train_accs.append(acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            early_stopping(val_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping.")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path))
        result_dir = os.path.join('./results', self.args.model_id, setting)
        os.makedirs(result_dir, exist_ok=True)

        df = pd.DataFrame({
            'Epoch': list(range(1, len(train_losses) + 1)),
            'Train Loss': train_losses,
            'Val Loss': val_losses,
            'Train Acc': train_accs,
            'Val Acc': val_accs
        })
        df.to_csv(os.path.join(result_dir, 'training_log.csv'), index=False)
        return self.model

    def evaluate(self, data_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                batch_x, batch_y = batch
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y)
                total_loss += loss.item()

                pred = outputs.argmax(dim=1)
                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)

                all_preds.append(pred.cpu())
                all_labels.append(batch_y.cpu())

        preds = torch.cat(all_preds).numpy()
        trues = torch.cat(all_labels).numpy()

        acc = accuracy_score(trues, preds)
        avg_loss = total_loss / len(data_loader)

        return acc, avg_loss

    def test(self, setting, test=0):
        best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
        if os.path.exists(best_model_path):
            try:
                self.model.load_state_dict(torch.load(best_model_path))
                print(f"✅ Loaded model from {best_model_path}")
            except:
                print(f"⚠️ Model not using state_dict.")

        test_data, test_loader = self._get_data('test')
        print(f"test set size: {len(test_data)} | num_classes: {self.args.c_out}")
        self.model.eval()

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch_x, batch_y = batch
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs = self.model(batch_x)

                pred = outputs.argmax(dim=1)

                correct += (pred == batch_y).sum().item()
                total += batch_y.size(0)
                all_preds.append(pred.cpu())
                all_labels.append(batch_y.cpu())

        acc = correct / total
        preds = torch.cat(all_preds).numpy()
        trues = torch.cat(all_labels).numpy()

        precision = precision_score(trues, preds, average='macro', zero_division=0)
        recall = recall_score(trues, preds, average='macro', zero_division=0)
        f1 = f1_score(trues, preds, average='macro', zero_division=0)

        print(f"\nTest Results:")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-score : {f1:.4f}")

        result_dir = os.path.join('./results', setting)
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, 'test_summary.txt'), 'w') as f:
            f.write(f"Acc: {acc:.4f}\nPre: {precision:.4f}\nRec: {recall:.4f}\nF1: {f1:.4f}")

        return acc
