import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchmetrics.classification import MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, MulticlassF1Score 

def train(model, 
          dataloader,
          optimizer,
          device,
          loss_mode,
          beta=0.999,
          gamma=2,
          class_freq=None):

    model.train() 

    # initialize loss
    train_epoch_loss = 0.0

    for batch in dataloader:

        # 1. load data
        batch = [tensor.to(device) for tensor in batch]
        train_src_act, train_src_time, train_tgt = batch
        # train_src_act shape: (batch_size, seq_len)
        # train_src_time shape: (batch_size, seq_len, 2)
        # train_tgt shape: (batch_size, seq_len)

        # 2. set the gradient to zero
        optimizer.zero_grad()

        # 3. run a forward pass and obtain predictions
        act_predictions = model(train_src_act, train_src_time) # shape: (batch_size, seq_len, num_act)

        # 4. calculate loss
        loss = loss_function(act_predictions, train_tgt, loss_mode, beta, gamma, class_freq)

        # 5. backpropagation
        loss.backward()
        optimizer.step()

        # 6. sum up losses from all batches
        train_epoch_loss += loss.item()

    # 7. divide total loss by number of batches to obain average loss
    avg_train_loss = train_epoch_loss / len(dataloader)

    return avg_train_loss


def validate(model, 
          dataloader,
          device,
          num_valid_class,
          loss_mode,
          beta=0.999,
          gamma=2,
          class_freq=None):

    model.eval()

    # initialize loss
    val_epoch_loss = 0.0
    all_act_labels = []
    all_act_tgt = []

    with torch.no_grad():

        for batch in dataloader:

            # 1. load data
            batch = [tensor.to(device) for tensor in batch]
            src_act, src_time, tgt = batch
            # src_act shape: (batch_size, seq_len)
            # src_time shape: (batch_size, seq_len, 2)
            # tgt shape: (batch_size, seq_len)

            # 2. run a forward pass and obtain predictions
            act_predictions = model(src_act, src_time) # shape: (batch_size, seq_len, num_act)
            act_labels = act_predictions.argmax(2) # shape: (batch_size, seq_len), int tensor

            # 3. calculate loss
            loss = loss_function(act_predictions, tgt, loss_mode, beta, gamma, class_freq)

            # 4. sum up losses from all batches
            val_epoch_loss += loss.item()

            # 5. accumulate predictions and targets for F1
            all_act_labels.append(act_labels.view(-1).detach())
            all_act_tgt.append(tgt.view(-1).detach())

    # 7. divide total loss by number of batches to obain average loss
    avg_val_loss = val_epoch_loss / len(dataloader)

    # 8. calculate f1 scores
    all_preds = torch.cat(all_act_labels) # 1D tensor
    all_targets = torch.cat(all_act_tgt) # 1D tensor
    accuracy, precision_macro, recall_macro, f1_macro = calculate_metrics(all_preds, all_targets, device, num_valid_class)

    return avg_val_loss, accuracy, precision_macro, recall_macro, f1_macro

def evaluate(model, 
          dataloader,
          device,
          num_valid_class):

    model.eval()

    # initialize loss
    all_act_labels = []
    all_act_tgt = []

    with torch.no_grad():

        for batch in dataloader:

            # 1. load data
            batch = [tensor.to(device) for tensor in batch]
            src_act, src_time, tgt = batch
            # src_act shape: (batch_size, seq_len)
            # src_time shape: (batch_size, seq_len, 2)
            # tgt shape: (batch_size,)

            # 2. run a forward pass and obtain predictions
            act_predictions = model(src_act, src_time) # shape: (batch_size, seq_len, num_act)
            act_labels = act_predictions.argmax(2) # shape: (batch_size, seq_len)

            # 3. get the predicted label
            prefix_lens = (src_act != 0).sum(dim=1)
            assert (prefix_lens > 0).all(), "Found at least one row in 'src_act' with all zeros."
            last_indices = prefix_lens - 1

            last_pred_label = act_labels.gather(1, last_indices.unsqueeze(1)).squeeze(1) # shape: (batch_size, )

            # 3. accumulate predictions and targets for F1
            all_act_labels.append(last_pred_label.detach())
            all_act_tgt.append(tgt)

    # 8. calculate f1 scores
    all_preds = torch.cat(all_act_labels)
    all_targets = torch.cat(all_act_tgt)
    accuracy, precision_macro, recall_macro, f1_macro = calculate_metrics(all_preds, all_targets, device, num_valid_class)

    return  accuracy, precision_macro, recall_macro, f1_macro, all_preds, all_targets

def loss_function(act_predictions, 
                  act_tgt,
                  loss_mode,
                  beta=0.999,
                  gamma=2,
                  class_freq=None):
    """
    Calculate cross-entropy loss for activity label prediction.

    Parameters
    ----------
    act_predictions: tensor
        shape: (batch_size, suffix_len, num_act)
    act_tgt: tensor
        shape: (batch_size, suffix_len)

    Returns
    -------  
    loss: tensor
        scalar, representing losses averaged over each loss element in the batch

    """

    if loss_mode == 'base':
        act_criterion = nn.CrossEntropyLoss(ignore_index=0)
    else:
        act_criterion = FocalLoss(beta=beta, gamma=gamma, class_freq=class_freq)

    act_predictions = act_predictions.view(-1, act_predictions.size(-1)) # shape: (batch_size * seq_length, num_act)
    act_tgt = act_tgt.view(-1) # shape: (batch_size * seq_length)

    loss = act_criterion(act_predictions, act_tgt)

    return loss

class FocalLoss(nn.Module):
    def __init__(self, beta, gamma, class_freq=None):
        super().__init__()
        self.beta = beta # hyperparameter for effective number hyperparameter
        self.gamma = gamma # hyperparameter for focal loss
        self.ce = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        if class_freq is not None:
            self.register_buffer('alpha', self.get_alpha(class_freq))
        else:
            self.register_buffer('alpha', None)

    def get_alpha(self, class_freq):

        effective_num = 1.0 - torch.pow(self.beta, class_freq.float())
        effective_num = torch.where(class_freq == 0, torch.ones_like(effective_num), effective_num)  # avoid zero division
        alpha = (1.0 - self.beta) / effective_num
        alpha = torch.where(class_freq == 0, torch.zeros_like(alpha), alpha) # alpha is 0 for class with frequency of 0
        alpha = alpha / alpha.sum() * (class_freq > 0).sum().float() # shape: (num_classes,)

        return alpha

    def forward(self, inputs, targets):

        ce_loss = self.ce(inputs, targets) 

        if ce_loss.dim() > 1:
            ce_loss = ce_loss.flatten() # shape: (batch_size * seq_len)

        if targets.dim() > 1:
            targets = targets.flatten()  # shape: (batch_size * seq_len)
        
        p_t = torch.exp(-ce_loss)  # shape: (batch_size * seq_len)
        
        focal_term = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss

        return loss.mean()

def calculate_metrics(act_labels, act_tgt, device, num_valid_class):

    Accuracy = MulticlassAccuracy(num_classes=num_valid_class, average='micro', ignore_index=0).to(device)
    accuracy = Accuracy(act_labels, act_tgt)

    Precision = MulticlassPrecision(num_classes=num_valid_class, average='macro', ignore_index=0).to(device)
    precision_macro = Precision(act_labels, act_tgt)

    Recall = MulticlassRecall(num_classes=num_valid_class, average='macro', ignore_index=0).to(device)
    recall_macro = Recall(act_labels, act_tgt)

    F1 = MulticlassF1Score(num_classes=num_valid_class, average='macro', ignore_index=0).to(device)
    f1_macro = F1(act_labels, act_tgt)

    return accuracy, precision_macro, recall_macro, f1_macro

class EarlyStopper:
    """
    Implements early stopping to terminate training when validation performance 
    stops improving.

    Inspired by: https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch

    Parameters
    ----------
    patience : int
        Number of epochs with no significant improvement in validation metrics 
        before stopping.

    Attributes
    ----------
    patience : int
        The number of epochs with no significant improvement in validation 
        metrics before stopping.
    counter : int
        The current number of consecutive epochs without improvement.
    min_val_loss : float
        The minimum validation loss observed.

    Methods
    -------
    early_stop(val_loss) -> bool
        Checks if training should be stopped.
    """
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        """
        Determine whether to stop training early based on validation loss.

        Parameters
        ----------
        val_loss : float
            Current epoch's validation loss.

        Returns
        -------
        bool
            True if training should stop early, False otherwise.
        """
        improvement = False
        
        if val_loss < (self.min_val_loss-self.delta):
            self.min_val_loss = val_loss
            improvement = True
        
        if improvement:
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True
        
        return False

def init_weights_uni(m):
    """
    Initialize model parameters uniformly in the range [-0.08, 0.08].

    Parameters
    ----------
    m : torch.nn.Module
        An instance of model.
    """
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.uniform_(param.data, -0.08, 0.08)

def init_weights_kaiming(m):
    for name, param in m.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            nn.init.kaiming_uniform_(param)

