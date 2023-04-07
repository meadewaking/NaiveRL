import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler
from d2l import torch as d2l
from model import Model
from collections import Counter


class Mydata(Dataset):
    def __init__(self):
        self.x = []
        self.y = []
        self.states = np.load('tools/states.npy')
        self.actions = np.load('tools/actions.npy')
        self.counter = Counter(self.actions)

    def __getitem__(self, idx):
        assert idx < self.actions.shape[0]
        self.x = self.states[idx]
        self.x = torch.tensor(self.x, dtype=torch.float)
        self.y = self.actions[idx]
        self.y = torch.tensor(self.y, dtype=torch.long)
        batch = (self.x, self.y)
        return batch

    def __len__(self):
        return self.actions.shape[0]


net = Model()


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.

    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net.pi(X), y), d2l.size(y))
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, num_epochs, lr, device):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print('training on', device)
    scaler = GradScaler()
    net.to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_epochs, eta_min=lr / 2)
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            # X = torch.tensor(X, dtype=torch.float)
            # y = torch.tensor(y, dtype=torch.long)
            X, y = X.to(device), y.to(device)
            with autocast():
                y_hat = net.pi(X)
                l = loss(y_hat, y)
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                metric.add(l * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 1) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
                # d2l.plt.savefig('model/test.jpg')
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
        if (epoch + 1) % 100 == 0:
            # torch.save(net, 'tools/manual_model_' + str(epoch + 1))
            pass
        scheduler.step()
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def Sample_weight():
    class_weight_dict = {}
    for sample_class in dataset.counter:
        class_sample_count = dataset.counter[sample_class]
        weight = sum(dataset.counter) / class_sample_count
        class_weight_dict[sample_class] = weight
    # class_weight_dict[1] *= 2
    weights = []
    for weight in dataset.actions[train_data.indices]:
        weights.append(class_weight_dict[weight] * 10)
    return weights


dataset = Mydata()
train_data, test_data = random_split(dataset=dataset, lengths=[int(0.9 * dataset.__len__()),
                                                               dataset.__len__() - int(0.9 * dataset.__len__())])
weights = Sample_weight()
sampler = WeightedRandomSampler(weights, num_samples=len(weights))
train_loader = DataLoader(train_data, batch_size=128, num_workers=8, drop_last=True, persistent_workers=True,
                          pin_memory=True, sampler=sampler)
test_loader = DataLoader(test_data, batch_size=512, shuffle=True, num_workers=8)
# dataloader = DataLoader(dataset, batch_size=512, num_workers=8, drop_last=True, persistent_workers=True,
#                         pin_memory=True, shuffle=True)
train(net, train_loader, test_loader, 600, 1e-5, 'cuda')
