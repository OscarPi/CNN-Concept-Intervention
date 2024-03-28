import torch

class LinearModels(torch.nn.Module):
    def __init__(self, n_filters, n_concepts):
        super().__init__()

        self.n_concepts = n_concepts
        self.l = torch.nn.Linear(n_filters, n_concepts)

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.l(x))

class Model(torch.nn.Module):
  def __init__(self, in_channels, width, height, n_classes):
    super().__init__()

    self.width = width
    self.height = height

    self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3), padding='same')
    self.postconv1 = torch.nn.Sequential(
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU())
    self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same')
    self.postconv2 = torch.nn.Sequential(
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU())
    self.conv3 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same')
    self.postconv3 = torch.nn.Sequential(
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU())
    self.conv4 = torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same')
    self.postconv4 = torch.nn.Sequential(
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU())
    self.out = torch.nn.Sequential(
      torch.nn.Flatten(),
      torch.nn.Linear(width * height * 16, n_classes))

  def forward(self, x, interventions=None):
    if interventions is not None:
      interventions = interventions[:, :, None, None].repeat(1, 1, self.width, self.height)

    f1 = self.postconv1(self.conv1(x))
    if interventions is not None:
      f1 += interventions[:, :16]

    f2 = self.postconv2(self.conv2(f1))
    if interventions is not None:
      f2 += interventions[:, 16:32]

    f3 = self.postconv3(self.conv3(f2))
    if interventions is not None:
      f3 += interventions[:, 32:48]

    f4 = self.postconv4(self.conv4(f3))
    if interventions is not None:
      f4 += interventions[:, 48:]
    
    f = torch.mean(torch.cat((f1, f2, f3, f4), dim=1), dim=(2, 3))

    return self.out(f4), f


class CBM(torch.nn.Module):
  def __init__(self, in_channels, width, height, n_concepts, n_classes):
    super().__init__()

    self.concept_encoder = torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Flatten(),
      torch.nn.Linear(width * height * 16, n_concepts)
    )

    self.label_predictor = torch.nn.Sequential(
      torch.nn.Linear(n_concepts, 128),
      torch.nn.LeakyReLU(),
      torch.nn.Linear(128, n_classes)
    )

  def forward(self, x, c=None, concept_intervention_mask=None):
    c_pred = self.concept_encoder(x)
    c_pred_sigmoidal = torch.nn.Sigmoid()(c_pred)

    if concept_intervention_mask is None:
      y = self.label_predictor(c_pred_sigmoidal)
    else:
      mask = torch.tensor(concept_intervention_mask).to(c.device)
      c_intervened = torch.where(mask, c, c_pred_sigmoidal)
      y = self.label_predictor(c_intervened)

    return (c_pred_sigmoidal, y)

def get_cem_c_extractor_arch(in_channels, width, height):
  def c_extractor_arch(output_dim):
    output_dim = output_dim or 128
    return torch.nn.Sequential(
      torch.nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3,3), padding='same'),
      torch.nn.BatchNorm2d(num_features=16),
      torch.nn.LeakyReLU(),
      torch.nn.Flatten(),
      torch.nn.Linear(width * height * 16, output_dim)
    )
  return c_extractor_arch
