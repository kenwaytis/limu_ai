import torch
from torch import nn
from d2l import torch as d2l

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def dropout_layer(X, dropout):
    assert 0 <= dropout <= 1
    if dropout == 1:
        return torch.zeros_like(X,device=device)
    if dropout == 0:
        return X
    mask = (torch.rand(X.shape,device=device) > dropout).float()
    return mask * X / (1.0 - dropout)


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

dropout1, dropout2 = 0.2, 0.5


class Net(nn.Module):
    def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2, is_training=True):
        super(Net, self).__init__()
        self.num_inputs = num_inputs
        self.training = is_training
        self.lin1 = nn.Linear(num_inputs, num_hiddens1).to(device)
        self.lin2 = nn.Linear(num_hiddens1, num_hiddens2).to(device)
        self.lin3 = nn.Linear(num_hiddens2, num_outputs).to(device)
        self.relu = nn.ReLU()

    def forward(self, X):
        H1 = self.relu(self.lin1(X.reshape((-1, self.num_inputs)).to(device)))
        if self.training == True:
            H1 = dropout_layer(H1, dropout1)
        H2 = self.relu(self.lin2(H1))
        if self.training == True:
            H2 = dropout_layer(H2, dropout2)
        out = self.lin3(H2)
        return out.to(device)


net = Net(num_inputs, num_outputs, num_hiddens1, num_hiddens2).to(device)

num_epochs, lr, batch_size = 10, 0.5, 256
loss = nn.CrossEntropyLoss(reduction='none')
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
trainer = torch.optim.SGD(net.parameters(), lr=lr)
d2l.gpu_train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
