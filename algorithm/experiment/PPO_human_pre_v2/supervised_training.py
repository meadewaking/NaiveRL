import os
from collections import Counter
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from config import config
from common import ensure_data_dir
from model import Model


class HumanDataset(Dataset):
    def __init__(self):
        self.states = np.load(config['states_file'], mmap_mode='r')
        self.actions = np.load(config['actions_file'])
        self.counter = Counter(self.actions.tolist())

    def __getitem__(self, idx):
        x = torch.as_tensor(np.array(self.states[idx]), dtype=torch.float)
        y = torch.as_tensor(self.actions[idx], dtype=torch.long)
        return x, y

    def __len__(self):
        return self.actions.shape[0]


def sample_weight(dataset, train_data):
    total = sum(dataset.counter.values())
    class_weight = {cls: total / count for cls, count in dataset.counter.items()}
    return [class_weight[int(dataset.actions[idx])] for idx in train_data.indices]


def evaluate_accuracy(net, data_iter, device):
    net.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in data_iter:
            x, y = x.to(device), y.to(device)
            pred = net.logits(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def train():
    ensure_data_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = HumanDataset()
    train_len = int(0.9 * len(dataset))
    test_len = len(dataset) - train_len
    train_data, test_data = random_split(dataset, lengths=[train_len, test_len])

    weights = sample_weight(dataset, train_data)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = DataLoader(train_data, batch_size=config['bc_batch_size'], sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=config['bc_batch_size'], shuffle=False)

    net = Model().to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(), lr=config['bc_learning_rate'], weight_decay=config['bc_weight_decay']
    )
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, config['bc_epoch'] + 1):
        net.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = net.logits(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), config['max_grad_norm'])
            optimizer.step()
            total_loss += loss.item() * x.shape[0]
            correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.numel()

        test_acc = evaluate_accuracy(net, test_loader, device) if test_len > 0 else 0.0
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        print("epoch :{}, loss : {:.4f}, train_acc : {:.4f}, test_acc : {:.4f}".format(
            epoch, train_loss, train_acc, test_acc
        ))

        if epoch % config['bc_save_every'] == 0 or epoch == config['bc_epoch']:
            torch.save(net.state_dict(), config['bc_model_file'])

    if not os.path.exists(config['bc_model_file']):
        torch.save(net.state_dict(), config['bc_model_file'])


if __name__ == '__main__':
    train()
