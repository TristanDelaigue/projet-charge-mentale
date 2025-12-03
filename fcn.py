import torch
import torch.nn as nn

class FCNBranch(nn.Module):
    def __init__(self, kernel_size):
        super(FCNBranch, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel_size)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.gap = nn.AvgPool1d(kernel_size=kernel_size)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.gap(x)
        return x

class FCNModel(nn.Module):
    def __init__(self, num_signals, kernel_size):
        super().__init__(self)

        self.branches = nn.ModuleList([FCNBranch(kernel_size) for i in range(num_signals)])

        self.fc = nn.Linear(in_features=128*num_signals, out_features=1)

    def forward(self, x_list):
        # x_list : liste de tenseurs [batch, 1, L] pour chaque signal
        outputs = []
        for i, branch in enumerate(self.branches):
            out = branch(x_list[i])
            outputs.append(out)
        # Concatenation
        x = torch.cat(outputs, dim=1)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        return x


def _prepare_tensor(x):
    # x may be tensor, numpy array, list or (tensor,) from dataloader collate
    if isinstance(x, (list, tuple)):
        x = x[0]
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    if x.dim() == 2:  # [batch, L] -> add channel dim
        x = x.unsqueeze(1)
    return x

def _prepare_label(y):
    if isinstance(y, (list, tuple)):
        y = y[0]
    if not torch.is_tensor(y):
        y = torch.tensor(y, dtype=torch.float32)
    return y



def epoch_train(_net, train_loader_l, y_train_loader, loss_func, optim):
    _net.train()
    tot_loss, n_samples=0,0

    for *x_batches, y_batch in zip(*train_loader_l, y_train_loader):
        optim.zero_grad()

        x_list = [_prepare_tensor(xb) for xb in x_batches]
        y = _prepare_label(y_batch)

        preds=_net(x_list)

        loss=loss_func(preds.squeeze(), y.float())

        loss.backward()
        optim.step()


        _net.eval()
        n_samples += y.size(0)
        tot_loss += loss.item() * y.size(0)
        _net.train()
    
    avg_loss = tot_loss / n_samples
    return avg_loss


def epoch_valid(_net, valid_loader_l, y_valid_loader, loss_func, optim):
    _net.eval()
    tot_loss, n_samples=0,0
    with torch.no_grad():
        for *x_batches, y_batch in zip(*valid_loader_l, y_valid_loader):
            x_list = [_prepare_tensor(xb) for xb in x_batches]
            y = _prepare_label(y_batch)

            preds = _net(x_list)

            loss = loss_func(preds.squeeze(), y.float())

            n_samples += y.size(0)
            tot_loss += loss.item() * y.size(0)

    _net.train()
    avg_loss = tot_loss / n_samples if n_samples > 0 else 0.0
    return avg_loss

def train(_net, train_loader_l, valid_loader_l, y_train_loader, y_test_loader, loss_func, optim, n_epochs):
    train_loss_list, valid_loss_list=[],[]

    for epoch in range(n_epochs):

        train_loss=epoch_train(_net, train_loader_l, y_train_loader, loss_func, optim)
        train_loss_list.append(train_loss)

        with torch.no_grad():
            valid_loss = epoch_valid(_net, valid_loader_l, y_test_loader, loss_func, optim)
            valid_loss_list.append(valid_loss)
    
        print(f'Epoch {epoch}:  train loss {train_loss}, valid loss {valid_loss}')
    return train_loss_list, valid_loss_list




    
