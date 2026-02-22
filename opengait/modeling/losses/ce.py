
import torch
import torch.nn.functional as F

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False, class_weights=None):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy
        # Handle class weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, logits, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)  # [n, 1]
        
        # Move class weights to same device as logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        
        # Scale logits
        scaled_logits = logits * self.scale  # [n, c, p]
        
        # Reshape for loss computation: [n, c, p] -> [n*p, c]
        scaled_logits_flat = scaled_logits.permute(0, 2, 1).contiguous().reshape(-1, c)  # [n*p, c]
        labels_flat = labels.repeat(1, p).reshape(-1)  # [n*p]
        
        if self.label_smooth:
            # Deterministic cross-entropy with label smoothing
            # Use log_softmax + nll_loss (both are deterministic)
            log_probs = F.log_softmax(scaled_logits_flat, dim=1)  # [n*p, c]
            
            # Label smoothing: (1-eps) * one_hot + eps / num_classes
            num_classes = c
            smooth_labels = torch.zeros_like(log_probs)  # [n*p, c]
            smooth_labels.fill_(self.eps / num_classes)
            smooth_labels.scatter_(1, labels_flat.unsqueeze(1), 1.0 - self.eps + self.eps / num_classes)
            
            # Compute loss: -sum(smooth_labels * log_probs)
            loss_per_sample = -(smooth_labels * log_probs).sum(dim=1)  # [n*p]
            
            # Apply class weights if provided
            if weight is not None:
                weight_per_sample = weight[labels_flat]  # [n*p]
                loss_per_sample = loss_per_sample * weight_per_sample
            
            loss = loss_per_sample.mean()
        else:
            # Deterministic cross-entropy without label smoothing
            # Use log_softmax + nll_loss (both are deterministic)
            log_probs = F.log_softmax(scaled_logits_flat, dim=1)  # [n*p, c]
            
            # Gather log probabilities for true labels
            labels_flat_long = labels_flat.long()  # [n*p]
            loss_per_sample = -log_probs.gather(1, labels_flat_long.unsqueeze(1)).squeeze(1)  # [n*p]
            
            # Apply class weights if provided
            if weight is not None:
                weight_per_sample = weight[labels_flat_long]  # [n*p]
                loss_per_sample = loss_per_sample * weight_per_sample
            
            loss = loss_per_sample.mean()
        
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info
