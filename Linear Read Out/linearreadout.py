# -*- coding: utf-8 -*-
"""

## Linear read-out

To examine distributed representations, another approach is to decode them with a linear readout. This can be done at each layer of the DBN, similar to clustering. We will use the hidden representations to classify FashionMNIST images with a simple linear classifier. This will help us evaluate the amount of information contained in each hidden representation. Before proceeding, let's define the class for the linear classifier.
"""

class LinearModel(torch.nn.Module):
  def __init__(self, layer_size):
    super().__init__()
    self.linear = torch.nn.Linear(layer_size, 10)

  def forward(self, x):
    return self.linear(x)

"""Next, we can create a linear classifier for every hidden layer of the DBN.



"""

layer_size = dbn_fmnist.rbm_layers[0].W.shape[1]
linear1 = LinearModel(layer_size).to(device)

layer_size = dbn_fmnist.rbm_layers[1].W.shape[1]
linear2 = LinearModel(layer_size).to(device)

layer_size = dbn_fmnist.rbm_layers[2].W.shape[1]
linear3 = LinearModel(layer_size).to(device)

layer_size = dbn_fmnist.rbm_layers[3].W.shape[1]
linear4 = LinearModel(layer_size).to(device)

"""Then, we can train the linear classifiers on the hidden representations from each layer using the actual labels of the FashionMNIST dataset as targets."""

def train_linear(linear, hidden_reprs, title, epochs = 1500):
  print(title)
  optimizer = torch.optim.SGD(linear.parameters(), lr=0.05)
  loss_fn = torch.nn.CrossEntropyLoss()

  losses = []
  for epoch in range(epochs):
    optimizer.zero_grad()
    predictions = linear(hidden_reprs).squeeze()
    targets = fmnist_tr.targets.reshape(predictions.shape[0])  # here are the labels
    loss = loss_fn(predictions, targets)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 150 == 0:
      print("epoch : {:3d}/{}, loss = {:.4f}".format(epoch + 1, epochs, loss))
  print()

  plt.plot(range(epochs), losses)
  plt.title(title)
  plt.xlabel("Epoch")
  plt.ylabel("Loss")
  plt.show()

  print("---------------------------------------------------------------------")

train_linear(linear1, hidden_repr_1, "Linear 1")
train_linear(linear2, hidden_repr_2, "Linear 2")
train_linear(linear3, hidden_repr_3, "Linear 3")
train_linear(linear4, hidden_repr_4, "Linear 4")

"""Let's now evaluate the trained linear readouts using the hidden representations computed on the *test* set:"""

hidden_repr_1_test = get_kth_layer_repr(fmnist_te.data, 0, device)
hidden_repr_2_test = get_kth_layer_repr(hidden_repr_1_test, 1, device)
hidden_repr_3_test = get_kth_layer_repr(hidden_repr_2_test, 2, device)
hidden_repr_4_test = get_kth_layer_repr(hidden_repr_3_test, 3, device)

# compute the classifier predictions:
predictions_test1 = linear1(hidden_repr_1_test)
predictions_test2 = linear2(hidden_repr_2_test)
predictions_test3 = linear3(hidden_repr_3_test)
predictions_test4 = linear4(hidden_repr_4_test)

def compute_accuracy(predictions_test, targets):
  predictions_indices = predictions_test.max(axis=1).indices  # convert probabilities to indices
  accuracy = (predictions_indices == targets).sum() / len(targets)
  return accuracy.item()

acc1 = compute_accuracy(predictions_test1, fmnist_te.targets)
acc2 = compute_accuracy(predictions_test2, fmnist_te.targets)
acc3 = compute_accuracy(predictions_test3, fmnist_te.targets)
acc4 = compute_accuracy(predictions_test4, fmnist_te.targets)

print("The linear read-out of the first layer performs an accuracy of {:.4f}.\n".format(acc1))
print("The linear read-out of the second layer performs an accuracy of {:.4f}.\n".format(acc2))
print("The linear read-out of the third layer performs an accuracy of {:.4f}.\n".format(acc3))
print("The linear read-out of the fourth layer performs an accuracy of {:.4f}.".format(acc4))
