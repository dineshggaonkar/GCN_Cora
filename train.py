import torch
import numpy as np
import config

from model import GCN


data = config.dataset[0]


node_classification_model = GCN(hidden_channels=config.hidden_channels, num_features=config.num_features,
                                num_classes=config.num_classes)


# Use GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
node_classification_model = node_classification_model.to(device)
data = data.to(device)

# Initialize Optimizer
learning_rate = 0.01
decay = 5e-4
optimizer = torch.optim.Adam(node_classification_model.parameters(),
                             lr=learning_rate,
                             weight_decay=decay)
# Define loss function (CrossEntropyLoss for Classification Problems with
# probability distributions)
criterion = torch.nn.CrossEntropyLoss()

def train():
    node_classification_model.train()
    optimizer.zero_grad()
    # Use all data as input, because all nodes have node features
    out = node_classification_model(data.x, data.edge_index)
    # Only use nodes with labels available for loss calculation --> mask
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


def test():
    node_classification_model.eval()
    out = node_classification_model(data.x, data.edge_index)
    # Use the class with highest probability.
    pred = out.argmax(dim=1)
    # Check against ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

min_train_loss = np.Inf
losses = []

for epoch in range(0, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    if round(losses[-1], 2) < min_train_loss:
        epochs_no_improve = 0
        min_train_loss = round(losses[-1], 2)
    else:
        epochs_no_improve += 1

    if epoch > 500 and epochs_no_improve == config.n_epochs_stop:
        print('Early stopping!')
        PATH = "./Gcn_model/Node_classif_model.pb"
        torch.save(node_classification_model.state_dict(), PATH)
        print("model saved")

        break
    else:
        continue

