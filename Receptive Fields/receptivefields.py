# -*- coding: utf-8 -*-
"""
## Visualizing receptive fields

Once the training is complete, we can plot the learned weights to visualize the **receptive fields** of each hidden neuron. This helps us understand which parts of an image activate the unit associated with a specific weight vector.

To reduce noise in the plots, we can apply a threshold on the learned weights. Additionally, since the values in different weights could have different ranges, we will use a `MinMaxScaler` from Scikit-Learn to ensure a meaningful comparison among the visualizations of different receptive fields.
"""

def get_weights(dbn, layer):
  # Gets the weights of the given layer
  return dbn.rbm_layers[layer].W.cpu().numpy()

def apply_threshold(weights, threshold=0):
  # Implements a given threshold to reduce the possible noise
  return weights * (abs(weights) > threshold)

def apply_min_max_scaler(learned_weights):
  # Applies the a MinMaxScaler to the weights of the layer
  original_shape = learned_weights.shape
  min_max_scaler = sklearn.preprocessing.MinMaxScaler()
  min_max_scaled_learned_weights = min_max_scaler.fit_transform(learned_weights.ravel().reshape(-1,1))
  min_max_scaled_learned_weights = min_max_scaled_learned_weights.reshape(original_shape)
  return min_max_scaled_learned_weights

def plot_layer_receptive_fields(weights, layer):
  # Plots some of the weights of the layers in order to visualize the receptive fields of the hidden neurons
  num_subplots = 36
  n_rows_cols = int(math.sqrt(num_subplots))
  fig, axes = plt.subplots(n_rows_cols, n_rows_cols, sharex=True, sharey=True, figsize=(10, 10))
  fig.suptitle('Receptive fields from the {} hidden layer'.format(layer), fontsize=16, y = 0.93)
  for i in range(num_subplots):
    row = i % n_rows_cols
    col = i // n_rows_cols
    axes[row, col].imshow(weights[i,:].reshape((28,28)), cmap=plt.cm.gray)  # here we select the weights we want to plot

w1 = get_weights(dbn_fmnist, layer=0)
w1 = apply_threshold(w1, 0.01) # a bigger threshold resulted in several useless subplots
w1 = apply_min_max_scaler(w1)
plot_layer_receptive_fields(w1.T, layer = 'first')

"""As we can see, one can still recognize familiar shapes in the receptive fields of the hidden units of the first layer. They seem to distinguish easily between three kind of shapes: the body clothes (t-shirts/tops, pullovers, dresses, trousers, coats and shirts), the shoes (sandals, sneakers and ankle boots) and the bags.

To visualize the weights in the second, third, and fourth hidden layers as images, we need to project each of the vectors into a space of dimensionality 784 (28x28) because they don’t have the same dimensionality as FashionMNIST images.
"""

w2 = get_weights(dbn_fmnist, layer=1)

w2 = apply_threshold(w2, 0.01)

w_product_12 = (w1 @ w2)  # here we do the projection (784, 200) @ (200, 400)
w_product_12 = apply_threshold(w_product_12, 0.01)
w_product_12 = apply_min_max_scaler(w_product_12)

plot_layer_receptive_fields(w_product_12.T, layer = 'second')

w3 = get_weights(dbn_fmnist, layer=2)

w3 = apply_threshold(w3, 0.01)

w_product_23 = (w_product_12 @ w3)  # here we do the projection (784, 400) x (400, 600)
w_product_23 = apply_threshold(w_product_23, 0.01)
w_product_23 = apply_min_max_scaler(w_product_23)

plot_layer_receptive_fields(w_product_23.T, 'third')

w4 = get_weights(dbn_fmnist, layer=3)

w4 = apply_threshold(w4, 0.01)

w_product_34 = (w_product_23 @ w4)  # here we do the projection (784, 600) x (600, 800)
w_product_34 = apply_threshold(w_product_34, 0.01)
w_product_34 = apply_min_max_scaler(w_product_34)

plot_layer_receptive_fields(w_product_34.T, 'fourth')

"""As we can see, the deeper the layer is, the more disentangled is its receptive field. The neurons in the deeper layers are less sensitive to the input features and more sensitive to the high-level features that are learned by the network. In the last layer, all features look very similar, which suggests that the network has learned a compact representation of the input data that is useful for classification."""
