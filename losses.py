from __future__ import print_function

import torch
import torch.nn as nn

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning Loss.
    Supports both supervised and SimCLR-style unsupervised contrastive loss.
    Paper: https://arxiv.org/pdf/2004.11362.pdf
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature  # scaling factor for similarity
        self.contrast_mode = contrast_mode  # use 'one' or 'all' views as anchors
        self.base_temperature = base_temperature  # base temp for scaling loss

    def forward(self, features, labels=None, mask=None):
        """
        Compute the supervised contrastive loss.

        Args:
            features: Tensor of shape [batch_size, n_views, embedding_dim]
            labels: Tensor of shape [batch_size] (optional)
            mask: Precomputed mask for positive pairs, shape [batch_size, batch_size] (optional)
        Returns:
            Scalar loss value
        """
        # Set device
        device = features.device

        # Check feature dimensions
        if len(features.shape) < 3:
            raise ValueError('`features` must be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            # Flatten the feature representation
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        # Build contrastive mask
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # Unsupervised case: only self-contrast (identity matrix)
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            # Supervised case: mask where entries are 1 if labels match
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Number of labels does not match number of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # Use provided mask
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # number of views (augmentations) per sample
        # Flatten features: shape becomes [batch_size * n_views, embedding_dim]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # Determine anchor features
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]  # use only the first view per sample
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature  # use all views as anchors
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown contrast_mode: {}'.format(self.contrast_mode))

        # Compute similarity (dot product scaled by temperature)
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # Subtract max for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Expand the mask for all anchors/views
        mask = mask.repeat(anchor_count, contrast_count)

        # Create a mask to zero out self-similarity
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask  # remove self-comparisons from mask

        # Compute log-softmax of positive logits
        exp_logits = torch.exp(logits) * logits_mask  # masked softmax numerator
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # log-softmax

        # Count number of positives per anchor
        mask_pos_pairs = mask.sum(1)
        # Avoid division by zero by setting to 1 where there are no positives
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

        # Average log_prob over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Final loss (scaled by temperature ratio)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()  # average across anchors

        return loss
