"""Padding-aware tensor helpers shared by score models and RL training."""
import torch


def last_valid_indices(mask: torch.Tensor) -> torch.Tensor:
    """Return the last unmasked index per row, for either padding side."""
    if mask.ndim != 2 or not mask.shape[0] or not mask.shape[1]:
        raise ValueError('Expected a nonempty 2-D attention mask')
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError('Attention mask must be binary')
    valid = mask.bool()
    if not valid.any(dim=1).all():
        raise ValueError('Every sequence must contain at least one valid token')
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~valid, -1).amax(dim=1)


def response_mask_from_lengths(lengths, width: int, device=None) -> torch.Tensor:
    """Build a right-padded response mask without guessing from logp values."""
    lengths = torch.as_tensor(lengths, device=device)
    if lengths.ndim != 1 or not lengths.numel() or lengths.dtype == torch.bool:
        raise ValueError('Expected a nonempty vector of integer response lengths')
    if not torch.all(lengths == lengths.long()):
        raise ValueError('Response lengths must be integers')
    lengths = lengths.long()
    if (lengths <= 0).any() or (lengths > width).any():
        raise ValueError('Response lengths must be positive and fit the response tensor')
    return torch.arange(width, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
