import torch
import torch.nn.functional as F

from .base import BaseLoss


class FocalLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, 
                 log_accuracy=False, gamma=2.0, class_weights=None):
        """
        Focal Loss: Focuses on hard-to-classify examples by down-weighting easy ones.
        
        Args:
            scale: Scale factor for logits (default: 16)
            label_smooth: Whether to use label smoothing (default: True)
            eps: Label smoothing epsilon (default: 0.1)
            loss_term_weight: Weight for this loss term (default: 1.0)
            log_accuracy: Whether to log accuracy (default: False)
            gamma: Focusing parameter, higher gamma focuses more on hard examples (default: 2.0)
            class_weights: Class weights for imbalanced datasets (default: None)
        """
        super(FocalLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy
        self.gamma = gamma
        
        # Handle class weights
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
        else:
            self.class_weights = None

    def forward(self, logits, labels):
        """
        Compute Focal Loss.
        
        Args:
            logits: [n, c, p] - model predictions (logits)
            labels: [n] - ground truth labels
        
        Returns:
            loss: focal loss value
            info: dictionary with loss and accuracy metrics
        """
        # logits: [n, c, p], labels: [n]
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)  # [n, 1]
        
        # Move class weights to same device as logits
        weight = None
        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        
        # Scale logits for better numerical stability
        scaled_logits = logits * self.scale  # [n, c, p]
        
        # Expand labels to match logits shape
        labels_expanded = labels.repeat(1, p)  # [n, p]
        
        # Get probabilities from softmax
        probs = F.softmax(scaled_logits, dim=1)  # [n, c, p]
        
        # Get probability of true class (pt)
        # probs: [n, c, p], labels_expanded: [n, p]
        # We need to gather along dimension 1 (class dimension)
        # Use the same approach as computing CE loss - flatten and process
        # Reshape to [n*p, c] for easier processing
        # Permute from [n, c, p] to [n, p, c], then flatten to [n*p, c]
        probs_permuted = probs.permute(0, 2, 1)  # [n, p, c]
        probs_flat = probs_permuted.contiguous().reshape(-1, c)  # [n*p, c] - make contiguous first
        labels_flat = labels_expanded.contiguous().reshape(-1).long()  # [n*p] - make contiguous first
        # Gather probabilities: for each of n*p items, get prob of its true class
        pt_flat = probs_flat.gather(1, labels_flat.unsqueeze(1)).squeeze(1)  # [n*p]
        pt = pt_flat.reshape(n, p)  # [n, p] - probability of true class
        
        # Compute standard cross entropy loss (per sample, per part)
        # Make tensors contiguous before reshaping to avoid view errors
        scaled_logits_flat = scaled_logits.permute(0, 2, 1).contiguous().reshape(-1, c)  # [n*p, c]
        labels_flat_ce = labels_expanded.contiguous().reshape(-1)  # [n*p]
        
        if self.label_smooth:
            ce_loss = F.cross_entropy(
                scaled_logits_flat, labels_flat_ce,
                weight=weight, label_smoothing=self.eps, reduction='none'
            ).reshape(n, p)  # [n, p]
        else:
            ce_loss = F.cross_entropy(
                scaled_logits_flat, labels_flat_ce,
                weight=weight, reduction='none'
            ).reshape(n, p)  # [n, p]
        
        # Compute focal weight: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma  # [n, p]
        
        # Apply focal weight to cross entropy loss
        focal_loss_per_sample = focal_weight * ce_loss  # [n, p]
        
        # Average over parts and samples
        focal_loss = focal_loss_per_sample.mean()
        
        # Update info dictionary
        self.info.update({'loss': focal_loss.detach().clone()})
        
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels_expanded).float().mean()
            self.info.update({'accuracy': accu})
        
        return focal_loss, self.info

