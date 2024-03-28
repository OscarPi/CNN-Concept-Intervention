import numpy as np
import torch

dataset_zip = np.load('dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')
p = np.load("permutation.npy")

p = p[dataset_zip["latents_classes"][p][:, 3] == 0]

x = torch.tensor(dataset_zip["imgs"][p])[:, None, :] / 255.0

loc = torch.tensor(dataset_zip["latents_classes"][:, (False, False, False, False, True, True)][p]) > 15
quadrant = loc[:, 0] * 2 + loc[:, 1]
shape = torch.tensor(dataset_zip["latents_classes"][:, (False, True, False, False, False, False)][p]).squeeze()
c = torch.stack((
    quadrant == 0,
    quadrant == 1,
    quadrant == 2,
    quadrant == 3,
    shape == 0,
    shape == 1,
    shape == 2), dim=1).float()
y = shape*4 + quadrant

l = len(x)
x_train, c_train, y_train = x[:int(0.6*l)], c[:int(0.6*l)], y[:int(0.6*l)]
x_val, c_val, y_val = x[int(0.6*l):int(0.75*l)], c[int(0.6*l):int(0.75*l)], y[int(0.6*l):int(0.75*l)]
x_test, c_test, y_test = x[int(0.75*l):], c[int(0.75*l):], y[int(0.75*l):]
train_dataset = torch.utils.data.TensorDataset(x_train, y_train, c_train)
val_dataset = torch.utils.data.TensorDataset(x_val, y_val, c_val)
test_dataset = torch.utils.data.TensorDataset(x_test, y_test, c_test)
train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=512)
val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=512)
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=512)
