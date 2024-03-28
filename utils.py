import torch
import cem.data.mnist_add as mnist_data_module
import numpy as np
from IPython.display import HTML, display

def progress(value, max):
  return HTML(
    f"<progress value='{value}' max='{max}', style='width: 100%'>"
    f"{value}"
    f"</progress>"
  )

def get_config(selected_digits, threshold_labels):
  batch_size = 2048
  return {
    "selected_digits": selected_digits,
    "num_operands": len(selected_digits),
    "threshold_labels": threshold_labels,
    "sampling_percent": 1,
    "sampling_groups": True,
    "train_dataset_size": 10000,
    "batch_size": batch_size,
    "num_workers": 0
  }

def load_dataset(config):
  train_dl, val_dl, test_dl, imbalance, (n_concepts, n_tasks, concept_map) = \
    mnist_data_module.generate_data(
      config=config,
      seed=42,
      output_dataset_vars=True,
      root_dir="/Users/oscar/Documents/ticks/2023-2024/Project/datasets/mnist",
    )

  return train_dl, val_dl, test_dl

loaded_datasets = {}
def dls(n_digits, selected_digits, threshold_labels=None):
  if (n_digits, selected_digits, threshold_labels) not in loaded_datasets:
    config = get_config(
      selected_digits=[list(selected_digits)] * n_digits,
      threshold_labels=threshold_labels,
    )
    loaded_datasets[(n_digits, selected_digits, threshold_labels)] = load_dataset(config)
  return loaded_datasets[(n_digits, selected_digits, threshold_labels)]

def train_dl(n_digits, selected_digits, threshold_labels=None):
  return dls(n_digits, selected_digits, threshold_labels=threshold_labels)[0]

def val_dl(n_digits, selected_digits, threshold_labels=None):
  return dls(n_digits, selected_digits, threshold_labels=threshold_labels)[1]

def test_dl(n_digits, selected_digits, threshold_labels=None):
  return dls(n_digits, selected_digits, threshold_labels=threshold_labels)[2]

def calculate_accuracy(dataloader, model, device="mps"):
  model = model.to(device)
  size = len(dataloader.dataset)
  model.eval()
  correct = 0
  with torch.no_grad():
    for X, y, _ in dataloader:
      X, y = X.to(device), y.to(device)
      results = model(X)
      pred = results[0]
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  return correct / size

def train(model, train_dl, val_dl, epochs=300, device="mps"):
  model = model.to(device)

  loss_fn = torch.nn.CrossEntropyLoss()

  optimiser = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=0.000004,
  )

  progress_bar = display(progress(0, epochs), display_id=True)

  for e in range(epochs):
    model.train()
    for X, y, _ in train_dl:
      X, y = X.to(device), y.to(device)

      results = model(X)
      y_pred = results[0]

      loss = loss_fn(y_pred, y)

      loss.backward()
      optimiser.step()
      optimiser.zero_grad()

    progress_bar.update(progress(e + 1, epochs))
    print(f"Accuracy: {calculate_accuracy(val_dl, model, device)}")

def calculate_concept_accuracy(dataloader, model, concept_model, device="mps"):
  model = model.to(device)
  concept_model = concept_model.to(device)
  size = len(dataloader.dataset) * concept_model.n_concepts
  model.eval()
  concept_model.eval()
  correct = 0
  individual_correct = np.zeros(concept_model.n_concepts)
  with torch.no_grad():
    for X, _, c in dataloader:
      X, c = X.to(device), c.to(device)
      _, f = model(X)
      c_pred = concept_model(f).cpu().detach().numpy() > 0.5
      c = c.cpu().detach().numpy()
      correct += (c == c_pred).sum()
      individual_correct += (c == c_pred).sum(axis=0)
  return correct / size, individual_correct / len(dataloader.dataset)

def train_concept_classifier(model, concept_classifier, train_dl, val_dl, epochs=300, device="mps"):
  model = model.to(device)
  concept_classifier = concept_classifier.to(device)
  concept_loss_fn = torch.nn.BCELoss()

  optimiser = torch.optim.Adam(
    concept_classifier.parameters(),
    lr=0.001,
    weight_decay=0.000004,
  )

  progress_bar = display(progress(0, epochs), display_id=True)

  for e in range(epochs):
    model.eval()
    concept_classifier.train()
    for X, _, c in train_dl:
      X, c = X.to(device), c.to(device)
      _, f = model(X)
      c_pred = concept_classifier(f)
      loss = concept_loss_fn(c_pred, c)
      loss.backward()
      optimiser.step()
      optimiser.zero_grad()
    
    progress_bar.update(progress(e + 1, epochs))
    concept_accuracy, individual_accuracies = calculate_concept_accuracy(val_dl, model, concept_classifier, device)
    print(f"Accuracy: {concept_accuracy}")
    print(f"Individual accuracies: {individual_accuracies}")

def calculate_accuracy_with_interventions(model, concept_classifier, test_dl, eps, concepts_to_intervene, intervene_incorrectly=False, device="mps"):
  model = model.to(device)
  concept_classifier = concept_classifier.to(device)
  size = len(test_dl.dataset)
  model.eval()
  concept_classifier.eval()
  correct = 0
  with torch.no_grad():
    for X, y, c in test_dl:
      X, y, c = X.to(device), y.to(device), c.to(device)
      if intervene_incorrectly:
          c = 1. - c
      batch_size = X.size(0)
      _, f = model(X)
      n_filters = f.size(1)
      c_pred = concept_classifier(f) > 0.5
      where_correct = (c == c_pred)[:, :, None].repeat(1, 1, n_filters)
      can_intervene = torch.full_like(where_correct, False)
      can_intervene[:, concepts_to_intervene] = True
      will_intervene = torch.logical_and(can_intervene, torch.logical_not(where_correct))
      filter_weights = concept_classifier.l.weight
      epsilon = torch.where(
          c[:, :, None].repeat(1, 1, n_filters) == 1.,
          eps * filter_weights[None, :, :].repeat(batch_size, 1, 1),
          -eps * filter_weights[None, :, :].repeat(batch_size, 1, 1))
      interventions = torch.where(will_intervene, epsilon, torch.zeros_like(epsilon)).sum(dim=1)
      y_pred, _ = model(X, interventions)
      correct += (y_pred.argmax(1) == y).type(torch.float).sum().item()
      
  return correct / size

def calculate_cbm_accuracies(dataloader, model, concept_intervention_mask=None, intervene_incorrectly=False, device="mps"):
  model.eval()
  correct_concept_predictions = 0
  total_concept_predictions = 0
  correct_task_predictions = 0
  total_task_predictions = 0
  with torch.no_grad():
    for X, y, c in dataloader:
      X, c, y = X.to(device), c.to(device), y.to(device)
      if intervene_incorrectly:
        c = 1. - c
      c_pred, y_pred = model(X, c, concept_intervention_mask)
      c_pred = (c_pred.cpu().detach().numpy() >= 0.5).astype(np.int32).flatten()
      c = c.cpu().detach().numpy().astype(np.int32).flatten()
      y_pred = y_pred.argmax(dim=-1).cpu().detach()
      y = y.cpu().detach()

      correct_concept_predictions += (c_pred == c).sum()
      total_concept_predictions += c.size

      correct_task_predictions += (y == y_pred).sum().item()
      total_task_predictions += y.size(dim=0)

  task_accuracy = correct_task_predictions / total_task_predictions
  concept_accuracy = correct_concept_predictions / total_concept_predictions
  return task_accuracy, concept_accuracy

def train_independent_cbm(model, dataloader, epochs=300, device="mps"):
  model = model.to(device)

  task_loss_fn = torch.nn.CrossEntropyLoss()
  concept_loss_fn = torch.nn.BCELoss()

  concept_encoder = model.concept_encoder.to(device)
  label_predictor = model.label_predictor.to(device)

  concept_encoder_optimiser = torch.optim.Adam(
    concept_encoder.parameters(),
    lr=0.001,
    weight_decay=0.000004,
  )
  label_predictor_optimiser = torch.optim.Adam(
    label_predictor.parameters(),
    lr=0.001,
    weight_decay=0.000004,
  )

  concept_encoder.train()
  label_predictor.train()

  progress_bar = display(progress(0, epochs), display_id=True)

  for e in range(epochs):
    for X, y, c in dataloader:
      X, c, y = X.to(device), c.to(device), y.to(device)

      c_pred = torch.nn.Sigmoid()(concept_encoder(X))
      y_pred = label_predictor(c)

      task_loss = task_loss_fn(y_pred, y)
      concept_loss = concept_loss_fn(c_pred, c)

      task_loss.backward()
      concept_loss.backward()
      concept_encoder_optimiser.step()
      label_predictor_optimiser.step()
      concept_encoder_optimiser.zero_grad()
      label_predictor_optimiser.zero_grad()

    progress_bar.update(progress(e + 1, epochs))

def calculate_cem_accuracies(dataloader, model, concept_intervention_mask=None, intervene_incorrectly=False, device="mps"):
  model.to(device)
  model.eval()
  correct_concept_predictions = 0
  total_concept_predictions = 0
  correct_task_predictions = 0
  total_task_predictions = 0
  with torch.no_grad():
    for X, y, c in dataloader:
      X, c, y = X.to(device), c.to(device), y.to(device)
      if intervene_incorrectly:
        c = 1. - c
      c_sem, c_pred, y_pred = model(X, c=c, intervention_idxs=concept_intervention_mask)
      c_sem = (c_sem.cpu().detach().numpy() >= 0.5).astype(np.int32).flatten()
      c = c.cpu().detach().numpy().astype(np.int32).flatten()
      y_pred = y_pred.argmax(dim=-1).cpu().detach()
      y = y.cpu().detach()

      correct_concept_predictions += (c_sem == c).sum()
      total_concept_predictions += c.size

      correct_task_predictions += (y == y_pred).sum().item()
      total_task_predictions += y.size(dim=0)

  task_accuracy = correct_task_predictions / total_task_predictions
  concept_accuracy = correct_concept_predictions / total_concept_predictions
  return task_accuracy, concept_accuracy
