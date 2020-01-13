import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Genotype
import Chromosome
import Population


def NN(mem, device=None):

    # load settings
    chromo_settings = mem.get_chromo()
    conv1win_dim = chromo_settings.get_conv1win_int()
    conv1chan_dim = chromo_settings.get_conv1chan_int()
    conv2win_dim = chromo_settings.get_conv2win_int()
    conv2chan_dim = chromo_settings.get_conv2chan_int()
    linear1_dim = chromo_settings.get_linear1_int()
    linear2_dim = chromo_settings.get_linear2_int()

    # load dataset and compute intermediate parameters
    data_set = mem.get_data_set()
    input_dims = {'MNIST': 28, 'CIFAR10': 32}
    in_chan_dims = {'MNIST': 1, 'CIFAR10': 3}
    input_dim = input_dims[data_set]
    in_chan_dim = in_chan_dims[data_set]

    post_conv1_dim = input_dim - conv1win_dim + 1
    post_max_pool1_dim = int(post_conv1_dim / 2)
    post_conv2_dim = post_max_pool1_dim - conv2win_dim + 1
    post_max_pool2_dim = int(post_conv2_dim / 2)
    post_flatten_dim = conv2chan_dim * post_max_pool2_dim * post_max_pool2_dim
    #print(chromo_settings)
    #print(input_dim, post_conv1_dim, post_max_pool1_dim, post_conv2_dim, post_max_pool2_dim)

    # load data
    if data_set is 'MNIST':
        train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
    elif data_set is 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_data = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform)
    else:
        raise Exception('Invalid input: dataset not availible in NN().')

    if device is not None:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=device[1])
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True, num_workers=device[1])
    else:
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=4, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)

    # define a net
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(in_chan_dim, conv1chan_dim, conv1win_dim)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(conv1chan_dim, conv2chan_dim, conv2win_dim)
            self.fc1 = nn.Linear(post_flatten_dim, linear1_dim)
            self.fc2 = nn.Linear(linear1_dim, linear2_dim)
            self.fc3 = nn.Linear(linear2_dim, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, post_flatten_dim)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    net = Net()
    if device is not None:
        net.to(device[0])

    # optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # train the network
    for epoch in range(1):
        running_loss = 0.0
        for data in train_loader:
            if device is not None:
                inputs, labels = data[0].to(device[0]), data[1].to(device[0])
            else:
                inputs, labels = data

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    #print('Finished Training')

    # test the network
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            if device is not None:
                inputs, labels = data[0].to(device[0]), data[1].to(device[0])
            else:
                inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print('Test acc (%d; %d): %.3f%%' % (linear1_neurons, linear2_neurons, 100 * correct / total))
    return 100 * correct / total

####################################################################################################

# run on GPU cluster
use_cluster = True
if use_cluster:
    device = [torch.device("cuda:0"), 1]
else:
    device = None

# initialize a (random) population
n_generations = 20
n_pop = 10
init_pop = Population.Pop(n_pop, data_set='CIFAR10')
curr_pop = init_pop

pop_history = []
all_acc_history = []
mean_acc_history = []

print('n_generations: %d\nn_pop: %d' % (n_generations, n_pop))

for g in range(n_generations):

    # get list of members
    curr_pop_list = curr_pop.to_list()

    # evaluate each members fitness
    print('\n Generation %d' % g)
    for i, mem in enumerate(curr_pop_list):
        # get accuracy
        acc = NN(mem, device=device)
        # store accuraacy
        mem.set_acc_(acc)
        # set trained status to true
        mem.set_status_(True)
        # print results
        chromo = mem.get_chromo()
        '''
        print('(%3d/%3d) [%3d, %3d, %3d, %3d, %3d, %3d]: %.1f%%' %
              (i+1, n_pop, chromo.get_conv1win_int(), chromo.get_conv1chan_int(),
               chromo.get_conv2win_int(), chromo.get_conv2chan_int(),
               chromo.get_linear1_int(), chromo.get_linear2_int(), acc))
        '''

    # store fully evaluated population
    pop_history.append(curr_pop)
    all_acc_history.append(curr_pop.get_all_acc())
    mean_acc_history.append(curr_pop.get_mean_acc())

    # create offspring population
    offspring_pop = curr_pop.offspring()

    # prepare to evaluate this generation
    curr_pop = offspring_pop

print(pop_history)
print(all_acc_history)
print(mean_acc_history)