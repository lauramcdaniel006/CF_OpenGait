import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class ContrastiveLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(ContrastiveLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        """
        Standard Contrastive Loss: 
        - Positive pairs (same class): L = D² (always minimize distance)
        - Negative pairs (different class): L = max(0, margin - D)² (only penalize if too close)
        
        Args:
            embeddings: [n, c, p] - feature embeddings
            labels: [n] - class labels
        
        Returns:
            loss: averaged contrastive loss
            info: dictionary with loss metrics
        """
        # embeddings: [n, c, p], labels: [n]
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()  # [n, c, p] -> [p, n, c]

        # Compute distance matrix for all pairs
        dist = self.ComputeDistance(embeddings, embeddings)  # [p, n, n]
        mean_dist = dist.mean((1, 2))  # [p]

        # Create positive/negative pairs
        pos_pairs, neg_pairs, num_pos_pairs_actual, num_neg_pairs_actual = self.CreatePairs(labels, dist)  # [p, n_pairs], [p, n_pairs], counts

        # Compute standard contrastive loss
        # Positive pairs: L = D² (always minimize distance - pull together)
        # Negative pairs: L = max(0, margin - D)² (only penalize if too close - push apart)
        pos_loss = pos_pairs ** 2  # [p, n_pos_pairs] - always minimize distance
        neg_loss = F.relu(self.margin - neg_pairs) ** 2  # [p, n_neg_pairs] - only penalize if distance < margin

        # Average losses
        pos_loss_avg = pos_loss.mean(1) if pos_loss.numel() > 0 else torch.zeros(dist.size(0), device=dist.device)
        neg_loss_avg = neg_loss.mean(1) if neg_loss.numel() > 0 else torch.zeros(dist.size(0), device=dist.device)
        
        # Total loss is average of positive and negative losses
        loss_avg = (pos_loss_avg + neg_loss_avg) / 2.0

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'pos_loss': pos_loss_avg.detach().clone(),
            'neg_loss': neg_loss_avg.detach().clone(),
            'mean_dist': mean_dist.detach().clone(),
            'num_pos_pairs': torch.tensor(num_pos_pairs_actual, device=dist.device),
            'num_neg_pairs': torch.tensor(num_neg_pairs_actual, device=dist.device)
        })

        return loss_avg, self.info

    def ComputeDistance(self, x, y):
        """
        Compute Euclidean distance between embeddings.
        
        Args:
            x: [p, n_x, c] - first set of embeddings
            y: [p, n_y, c] - second set of embeddings
        
        Returns:
            dist: [p, n_x, n_y] - pairwise distances
        """
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def CreatePairs(self, labels, dist):
        """
        Create positive and negative pairs from distance matrix.
        
        Args:
            labels: [n] - class labels
            dist: [p, n, n] - distance matrix
        
        Returns:
            pos_pairs: [p, n_pos_pairs] - distances for positive pairs
            neg_pairs: [p, n_neg_pairs] - distances for negative pairs
        """
        # Create mask for positive pairs (same class)
        labels_expanded = labels.unsqueeze(1)  # [n, 1]
        matches = (labels_expanded == labels_expanded.T).bool()  # [n, n]
        
        # Mask out diagonal (self-pairs)
        n = labels.size(0)
        mask = torch.eye(n, dtype=torch.bool, device=labels.device)
        matches = matches & (~mask)  # Remove self-pairs
        
        # Create mask for negative pairs (different class)
        neg_matches = ~matches & (~mask)  # Different class and not self
        
        p = dist.size(0)
        
        # Extract positive pair distances
        pos_distances = []
        neg_distances = []
        
        for i in range(p):
            # Get positive pairs (upper triangle to avoid duplicates)
            pos_upper = torch.triu(matches, diagonal=1)  # [n, n]
            pos_indices = pos_upper.nonzero(as_tuple=False)  # [n_pos_pairs, 2]
            
            if pos_indices.size(0) > 0:
                # Extract distances for positive pairs
                pos_dists = dist[i][pos_indices[:, 0], pos_indices[:, 1]]  # [n_pos_pairs]
                pos_distances.append(pos_dists)
            else:
                pos_distances.append(torch.tensor([], device=dist.device))
            
            # Get negative pairs (upper triangle to avoid duplicates)
            neg_upper = torch.triu(neg_matches, diagonal=1)  # [n, n]
            neg_indices = neg_upper.nonzero(as_tuple=False)  # [n_neg_pairs, 2]
            
            if neg_indices.size(0) > 0:
                # Extract distances for negative pairs
                neg_dists = dist[i][neg_indices[:, 0], neg_indices[:, 1]]  # [n_neg_pairs]
                neg_distances.append(neg_dists)
            else:
                neg_distances.append(torch.tensor([], device=dist.device))
        
        # Stack all parts
        max_pos = max([p.size(0) for p in pos_distances]) if pos_distances and any(p.size(0) > 0 for p in pos_distances) else 0
        max_neg = max([n.size(0) for n in neg_distances]) if neg_distances and any(n.size(0) > 0 for n in neg_distances) else 0
        
        if max_pos > 0:
            pos_pairs = torch.stack([
                F.pad(p, (0, max_pos - p.size(0)), value=0) if p.size(0) < max_pos else p[:max_pos]
                for p in pos_distances
            ])  # [p, max_pos]
        else:
            pos_pairs = torch.zeros(p, 1, device=dist.device)
        
        if max_neg > 0:
            neg_pairs = torch.stack([
                F.pad(n, (0, max_neg - n.size(0)), value=0) if n.size(0) < max_neg else n[:max_neg]
                for n in neg_distances
            ])  # [p, max_neg]
        else:
            neg_pairs = torch.zeros(p, 1, device=dist.device)
        
        # Return actual counts (before padding)
        num_pos_pairs_actual = max([p.size(0) for p in pos_distances]) if pos_distances and any(p.size(0) > 0 for p in pos_distances) else 0
        num_neg_pairs_actual = max([n.size(0) for n in neg_distances]) if neg_distances and any(n.size(0) > 0 for n in neg_distances) else 0
        
        return pos_pairs, neg_pairs, num_pos_pairs_actual, num_neg_pairs_actual

