import torch
from torch import nn

x = torch.tensor([1, 2])
y = torch.tensor([2, 3])
z = torch.tensor([2, 4])

t1 = [0, 1]
t2 = [1, 0]


sample1 = [(x, y), t1]
sample2 = [(y, z), t2]

######
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.enc = nn.Linear(in_features=2, out_features=2)

    def forward(self, state):
        state = self.enc(state)
        return state

class Inv(nn.Module):
    def __init__(self):
        super(Inv, self).__init__()
        self.inv = nn.Linear(in_features=4, out_features=2)
    def forward(self, state):
        state = self.inv(state)
        return state

def chain( *iterables):
    for it in iterables:
        for element in it:
            yield element

encoder_nn = Encoder()
inverse_nn = Inv()
params = chain(*[encoder_nn.parameters(), inverse_nn.parameters()])
optimizer = torch.optim.RMSprop(params, lr=0.0001)
loss_fn = nn.CrossEntropyLoss()

for i in [sample1, sample2]:
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    emb_0 = encoder_nn(i[0][0])
    emb_1 = encoder_nn(i[0][1])
    logits = inverse_nn(emb_0, emb_1)
    # Compute the loss and its gradients
    loss = loss_fn(logits, i[1])
    print(loss)
    loss.backward()
    # Adjust learning weights
    optimizer.step()
    print()


optimizer.zero_grad()
a = sample1[1]
state = encoder_nn(sample1[0][0])

for i in [sample2]:
    s = 0
    # Zero your gradients for every batch!
    optimizer.zero_grad()
    next_state = encoder_nn(i[0][0])
    logits = inverse_nn(state, next_state)
    # Compute the loss and its gradients
    loss = loss_fn(logits, a)
    loss.backward()
    a = i[1]
    # Adjust learning weights
    optimizer.step()
    state = next_state

