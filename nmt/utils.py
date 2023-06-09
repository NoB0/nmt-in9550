"""Utils functions."""
import torch
import math


def count_parameters(model):
    """Counts the number of trainable parameters in a model.

    Args:
        model: A PyTorch model.

    Returns:
        The number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rate(step: int, warmup_steps: int, d_model: int) -> float:
    """Learning rate scheduler from the Attention is All You Need paper."""
    if step == 0:
        step = 1
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def cosine_schedule_with_warmup(
    optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            min_factor,
            min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)),
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def generate_square_subsequent_mask(sz):
    from nmt.main import DEVICE

    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, pad_idx=0):
    from nmt.main import DEVICE

    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
