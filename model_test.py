from main import Net, learning_rate, momentum, train_loader, test_loader, network
import torch.optim as optim
import torch

continued_network = Net()
continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

network_state_dict = torch.load()
continued_network.load_state_dict(network_state_dict)

optimizer_state_dict = torch.load()
continued_optimizer.load_state_dict(optimizer_state_dict)
