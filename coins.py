### This attempts to both learn the distribution and the uncertainty around the predictions
### The uncertainty is the error distribution given x, and is assumed to have mean 0
### For a random batch, the batch error distribution is an average over the sample's error
### distributions. So random batch statistics will likely be underpowered making estimation
### hard. Clustering samples makes it easier to discern the error distributions.
### We could learn a parametric error distribution, or represent it with hierarchical priors
### or we could attempt to learn it (GAN on mixture distributions, hard, probably).
### We will show it in simple cases and scale up. A clustering module is doable with NNs.

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
import torch.nn.functional as F
import pdb
import datetime as dt

from data_util import load_mnist


def build_model(input_dim, output_dim):
    # For CEL, we don't need the softmax layer here since CrossEntropyLoss already
    # uses it internally.
    model = torch.nn.Sequential()
    model.add_module("linear",
                     torch.nn.Linear(input_dim, input_dim, bias=False))
    model.add_module("lrelu",
                     torch.nn.LeakyReLU())
    model.add_module("linear2",
                     torch.nn.Linear(input_dim, output_dim, bias=False))
    # model.add_module("sigmoid",
    #                  torch.nn.Sigmoid())
    return model.cuda()


def train(model, loss, optimizer, x_val, y_val):
    x, y = x_val, y_val
    # x = Variable(x_val, requires_grad=False)
    # y = Variable(y_val, requires_grad=False)

    # Reset gradient
    optimizer.zero_grad()

    # pdb.set_trace()
    # Forward
    fx = model.forward(x)
    output = loss.forward(fx, y)

    # Backward
    output.backward(retain_graph=True)

    # Update parameters
    optimizer.step()

    return output.data[0], fx


def train_random_effects(model, loss, optimizer, x, y, yhat, z, lambda_kld=0.1):
    optimizer.zero_grad()
    re_y = y - yhat
    # re_pred, re_kld_loss, newz = model(x, re_y, z)

    # pdb.set_trace()
    xz = x.unsqueeze(0).expand(z.size()[2], -1, -1).contiguous().view(-1, x.size()[1])  # (BR)xX
    yz = re_y.unsqueeze(0).expand(z.size()[2], -1, -1).contiguous().view(-1, re_y.size()[1])  # (BR)xY
    re_pred, re_kld_loss, newz = model(xz, yz,
                                       z.transpose(1,2).contiguous().view(-1, z.size()[1]))  # (BR)xY
    mean0_loss = re_pred.view(y.size()[0], -1)
    
    re_loss = loss(re_pred, yz)  # + re_pred.mean().pow(2)
    # re_loss = (torch.sort(re_pred,0)[0] - torch.sort(re_y,0)[0]).pow(2).mean()  # todo cdf->gap (KS)
    mixed_loss = re_loss + lambda_kld * re_kld_loss + mean0_loss.mean(0).pow(2).mean()
    # todo penalize according to sampling distribution under CLT not squared mean loss
    mixed_loss.backward()
    optimizer.step()
    return re_loss, re_kld_loss


def predict(model, x_val):
    x = x_val
    output = model.forward(x)
    return output.data.cpu().numpy().argmax(axis=1)


class Random_Effects(nn.Module):
    def __init__(self, x_size, y_size, z_size, hidden_size, output_activation):
        super(Random_Effects, self).__init__()
        self.map1 = nn.Linear(x_size + y_size, hidden_size)
        self.map1b = nn.Linear(hidden_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, 2*z_size)
        self.z_size = z_size
        self.map3 = nn.Linear(z_size, hidden_size)
        self.map4 = nn.Linear(hidden_size + x_size, hidden_size)
        self.map5 = nn.Linear(hidden_size, hidden_size)
        self.map5b = nn.Linear(hidden_size, hidden_size)
        self.map5c = nn.Linear(hidden_size, y_size)
        self.to_out = output_activation  # nn.Linear(hidden_size, output_size)
        
    def forward(self, x, y, z):
        t = F.leaky_relu(self.map1(torch.cat((x,y),1)))
        t = F.leaky_relu(self.map1b(t))
        # pdb.set_trace()
        self.means, self.logvar = torch.split(self.map2(t), self.z_size, 1)
        kld = (-0.5 * (1 + self.logvar - self.means.pow(2) - self.logvar.exp()).sum(1)).mean()
        newz = (self.logvar/2).exp() * z + self.means  # if kld is 0 this is still N(0,1)
        t = F.leaky_relu(self.map3(newz))
        t = F.relu(self.map4(torch.cat((t, x),1)))
        t = F.leaky_relu(self.map5(t))
        t = F.relu(self.map5b(t))
        t = self.map5c(t)
        t = self.to_out(t)
        return t, kld, newz

    def decoder(self, x, z):
        t = F.leaky_relu(self.map3(z))
        t = F.relu(self.map4(torch.cat((t, x),1)))
        t = F.leaky_relu(self.map5(t))
        t = F.relu(self.map5b(t))
        t = self.map5c(t)
        t = self.to_out(t)
        return t


class Random_Effects_Loss(nn.Module):
    def __init__(self):
        super(Random_Effects_Loss, self).__init__()

    def forward(self, hat_p, p):
        # The random effects loss really should be a distribution loss (but these are expensive)
        return (hat_p - p).pow(2).sum(1).mean()  # MSELoss
    

# def main():
torch.manual_seed(1)

### MNIST
# n_classes = 10
# trX, teX, trY, teY = load_mnist(onehot=False)
# trX = Variable(torch.from_numpy(trX).float()).cuda()
# teX = Variable(torch.from_numpy(teX).float()).cuda()
# trY = Variable(torch.from_numpy(trY).long()).cuda()
# trYwide = Variable(torch.FloatTensor(trY.size()[0], n_classes)).zero_().cuda().\
#     scatter_(1,trY.unsqueeze(1),1)

### Coin flips
# n_classes = 1
# n_coins, n_samples = 100, 1000
# p_heads = torch.FloatTensor(n_coins).uniform_()
# trX, teX = (torch.randint(n_coins, (n_samples,1)).long(),
#             torch.randint(n_coins, (n_samples,1)).long())
# trY, teY = (torch.bernoulli(p_heads[trX]).cuda(),
#             torch.bernoulli(p_heads[teX]).cuda())
# trYwide = trY
# trX, teX = (torch.FloatTensor(n_samples, n_coins).zero_().scatter_(1,trX,1).cuda(),
#             torch.FloatTensor(n_samples, n_coins).zero_().scatter_(1,teX,1).cuda())


### Coin flips, Poissons, and Normals
n_classes = 1
n_coins, n_samples = 5, 1000
p_heads = torch.FloatTensor(n_coins).uniform_()
bpn = torch.multinomial(torch.FloatTensor((0.4, 0.0, 0.6)),n_samples,replacement=True)
trX, teX = torch.FloatTensor(n_samples, 2), torch.FloatTensor(n_samples, 2)
trY, teY = torch.FloatTensor(n_samples, 1), torch.FloatTensor(n_samples, 1)
for i in range(n_samples):
    trX[i] = torch.FloatTensor((torch.randint(n_coins, (1,1)), bpn[i]))
    teX[i] = torch.FloatTensor((torch.randint(n_coins, (1,1)), bpn[i]))
    if bpn[i] == 0:
        trY[i] = torch.bernoulli(p_heads[trX[i,0].long()]).float()
        teY[i] = torch.bernoulli(p_heads[teX[i,0].long()]).float()
    elif bpn[i] == 1:
        trY[i] = torch.distributions.poisson.Poisson((p_heads[trX[i,0].long()]*10)).\
            sample().float()
        teY[i] = torch.distributions.poisson.Poisson((p_heads[teX[i,0].long()]*10)).\
            sample().float()
    else:
        trY[i] = torch.normal(10*p_heads[trX[i,0].long()]-5, 10*p_heads[trX[i,0].long()])
        teY[i] = torch.normal(10*p_heads[teX[i,0].long()]-5, 10*p_heads[teX[i,0].long()])
trX, trY, teX, teY = trX.cuda(), trY.cuda(), teX.cuda(), teY.cuda()
trYwide = trY

noise_size = 64
noise_replicates = 32

### Main model
n_examples, n_features = trX.size()
model = build_model(n_features, n_classes)
# loss = torch.nn.CrossEntropyLoss(size_average=True)
# loss = torch.nn.BCELoss(size_average=True)
loss = torch.nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer_main = optim.Adam(model.parameters())

### Random effects model
hidden_size = 32
# output_activation = nn.Softmax(dim=1)
# output_activation = nn.Tanh()
output_activation = nn.Linear(1,1)
re_model = Random_Effects(trX.size()[1], n_classes, noise_size, hidden_size, output_activation).cuda()
re_optimizer = optim.Adam(re_model.parameters())
re_loss = Random_Effects_Loss()
batch_size = 64
epochs = 3000
print_interval = 100

z = torch.FloatTensor(torch.Size((batch_size,noise_size,noise_replicates))).cuda()

### Do learning of {main, random effects}
for i in range(epochs):
    cost, re_cost, kl_cost = 0., 0., 0.
    num_batches = n_examples // batch_size
    for k in range(num_batches):
        start, end = k * batch_size, (k + 1) * batch_size
        cost_batch, yhat_batch = train(model, loss, optimizer_main,
                                       trX[start:end], trY[start:end])
        cost += cost_batch
        z = z.normal_()
        # if i == 950:
        #     pdb.set_trace()
        re_costs = train_random_effects(re_model, re_loss, re_optimizer,
                                        trX[start:end], trYwide[start:end],
                                        yhat_batch, z)
        re_cost += re_costs[0]
        kl_cost += re_costs[1]
    # predY = predict(model, teX) # for categorical
    predY = model(teX)

    if (i % print_interval) == 0:
        print("Epoch %d, cost = %f, acc = %.2f%%; re = %.3f"
              % (i + 1,
                 cost / num_batches,
                 ((predY > 0.5).float() == teY).sum().float()/predY.size()[0],
                 re_cost/num_batches))



### Diagnostics  # (for coins only)
### 1. Point estimate
print('\nPr(h) ', p_heads)
print('Train\nyhat: ', model(trX[:10]).squeeze(), '\ny   : ', p_heads[trX[:10].max(1)[1]])
print('Test\nyhat: ', model(teX[:10]).squeeze(), '\ny   : ', p_heads[teX[:10].max(1)[1]])

### 2. Error distribution
print('\nre_hat: ', -re_model.decoder(trX[:10], z.normal_()[:10,:,0]).squeeze(),
      '\ny-yhat: ', (trY[:10] - model(trX[:10])).squeeze())


nowtime = str(dt.datetime.now())
# Plot across the full distribution
plt.figure()
plt.hist(re_model.decoder(teX[:1000],
                          torch.FloatTensor(1000,z.size()[1]).cuda().normal_()[:1000]).
         squeeze().data.cpu().numpy(),30, alpha=0.5)
plt.hist((teY[:1000] - model(teX[:1000])).squeeze().data.cpu().numpy(), 30, facecolor='r', alpha=0.3)
plt.savefig('outputs/' + nowtime + 'fulldists.png')

# Plot across individual samples # verify you get distribution despite single label
n_plots = np.minimum(n_coins,9)
plt.figure(figsize=(5,n_plots*2))
for i in range(n_plots):
    plt.subplot(str(n_plots) + '1' + str(i+1))
    # plt.hist(re_model.decoder(torch.diag(torch.ones(n_coins))[i,:].expand_as(teX[:500]).cuda(),
    #                            torch.FloatTensor(500,z.size()[1]).cuda().normal_()[:500]).
    #          squeeze().data.cpu().numpy(),30, alpha=0.5)
    # plt.vlines([-p_heads[i], 1-p_heads[i]], 0,[500.*(1-p_heads[i]), 500.*(p_heads[i])], color='r')
    plt.hist(re_model.decoder(teX[i].unsqueeze(0).expand_as(teX[:1000]),
                              torch.FloatTensor(1000,z.size()[1]).cuda().normal_()[:1000]).
             squeeze().data.cpu().numpy(),50, alpha=0.5)
    plt.ylabel(('Bern','Pois','Norm')[teX[i,1].long().cpu().numpy()] +
               str(np.round(p_heads[i].data.cpu().numpy(),2)))
    plt.xlim(-10,10)
plt.savefig('outputs/' + nowtime + 'eachdist.png')



# if __name__ == "__main__":
# main()
