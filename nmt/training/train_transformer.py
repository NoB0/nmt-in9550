"""Training functions for Transformer-based Seq2Seq model."""

import logging
from nmt.dataset.dataset import collate_fn
from nmt.models.transformer import Seq2SeqTransformer
from nmt.utils import create_mask
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from timeit import default_timer as timer

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def save_transformer(model: Seq2SeqTransformer, path: str) -> None:
    """Saves a Transformer-based Seq2Seq model.

    Args:
        model: Transformer-based Seq2Seq model.
        path: Path to save model to.
    """
    logger.info(f"Saving model to {path}...")
    torch.save(
        {
            "vocab_size": model.src_vocab_size,
            "emb_size": model.emb_size,
            "hidden_size": model.dim_feedforward,
            "num_encoder_layers": model.num_encoder_layers,
            "num_decoder_layers": model.num_decoder_layers,
            "num_heads": model.nhead,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def train_transformer_epoch(
    model: Seq2SeqTransformer,
    optimizer,
    loss_fn,
    pad_id: int,
    training_dataset,
    device,
    batch_size: int,
) -> float:
    """Trains a single epoch for Transformer-based Seq2Seq model.

    Args:
        model: Transformer-based Seq2Seq model.
        optimizer: Optimizer.
        loss_fn: Loss function.
        training_dataset: Training dataset.
        pad_id: Padding token id.
        device: Device to use.
        batch_size: Batch size.

    Returns:
        Epoch loss.
    """
    model.train()
    losses = 0
    train_iter = training_dataset
    train_dataloader = DataLoader(
        train_iter,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
    )

    progress_bar = tqdm(range(len(train_dataloader)))
    for src, tgt in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()
        # print(loss.item())
        progress_bar.update(1)

    return losses / len(list(train_dataloader))


def evaluate_transformer(
    model: Seq2SeqTransformer,
    loss_fn,
    valid_dataset,
    batch_size: int,
    device,
    pad_id: int,
):
    """Evaluates a Transformer-based Seq2Seq model.

    Args:
        model: Transformer-based Seq2Seq model.
        loss_fn: Loss function.
        valid_dataset: Validation dataset.
        batch_size: Batch size.
        device: Device to use.
        pad_id: Padding token id.

    Returns:
        Validation loss.
    """
    model.eval()
    losses = 0

    val_iter = valid_dataset
    val_dataloader = DataLoader(
        val_iter,
        batch_size=batch_size,
        collate_fn=lambda batch: collate_fn(batch, pad_id),
    )

    for src, tgt in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, tgt_input
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            src_padding_mask,
        )

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))


def training_loop_transformer(
    model: Seq2SeqTransformer,
    optimizer,
    loss_fn,
    training_dataset,
    valid_dataset,
    device,
    batch_size: int,
    n_epochs: int,
    patience: int,
    pad_id: int,
    model_path: str,
) -> None:
    """Training loop.

    Args:
        model: Transformer-based Seq2Seq model.
        optimizer: Optimizer.
        loss_fn: Loss function.
        n_epochs: Number of epochs.
        patience: Patience.
        training_dataset: Training dataset.
        valid_dataset: Validation dataset.
        device: Device to use.
        batch_size: Batch size.
        pad_id: Padding token id.
        model_path: Path to save model to.
    """
    best_valid_loss = float("inf")
    patience_counter = 0

    logger.info("***** Running training *****")
    logger.info(f"  Model = {model.__class__.__name__}")
    logger.info(f"  Num Epochs = {n_epochs}")
    logger.info(f"  Patience = {patience}")

    for epoch in tqdm(range(1, n_epochs + 1)):
        start_time = timer()
        train_loss = train_transformer_epoch(
            model, optimizer, loss_fn, pad_id, training_dataset, device, batch_size
        )
        end_time = timer()
        val_loss = evaluate_transformer(
            model, loss_fn, valid_dataset, batch_size, device, pad_id
        )
        logger.info(
            (
                f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "
                f"Epoch time = {(end_time - start_time):.3f}s"
            )
        )

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= 3:
            logger.info("Early stopping.")
            break

    save_transformer(model, model_path)
