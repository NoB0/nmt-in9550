"""Training functions for RNN-based Seq2Seq model."""

import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from timeit import default_timer as timer
import torch
from nmt.dataset.dataset import CollateFunctor
from nmt.models.rnn import Seq2SeqModelRNN

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def save_rnn(model: Seq2SeqModelRNN, path: str) -> None:
    """Saves a RNN-based Seq2Seq model.

    Args:
        model: RNN-based Seq2Seq model.
        path: Path to save model to.
    """
    logger.info(f"Saving model to {path}...")
    torch.save(
        {
            "vocab_size": model.vocab_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "decoder_dropout": model.decoder_dropout,
            "encoder_dropout": model.encoder_dropout,
            "model_state_dict": model.state_dict(),
        },
        path,
    )


def train_rnn_epoch(
    model,
    optimizer,
    lr_scheduler,
    loss_fn,
    training_dataset,
    pad_id: int,
    batch_size,
    device,
    teacher_forcing_ratio=0.5,
) -> float:
    """Trains a single epoch for RNN-based Seq2Seq model.

    Args:
        model: RNN-based Seq2Seq model.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        training_dataset: Training dataset.
        pad_id: Padding token id.
        batch_size: Batch size.
        device: Device to use.
        teacher_forcing_ratio: Teacher forcing ratio. Defaults to 0.5.

    Returns:
        Epoch loss.
    """
    model.train()

    losses = 0
    train_iter = training_dataset
    train_dataloader = DataLoader(
        train_iter, batch_size=batch_size, collate_fn=CollateFunctor(pad_id)
    )

    progress_bar = tqdm(range(len(train_dataloader)))
    for src, _, tgt, _, _, _ in train_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        encoder_hidden, encoder_cell = model.encoder.init_hidden(src.shape[0])

        optimizer.zero_grad()

        target_input_ids = tgt[:, :-1]
        output = model(
            src,
            encoder_hidden,
            encoder_cell,
            target_input_ids,
            teacher_forcing_ratio,
        )
        output_dim = output.shape[-1]

        output = output.transpose(1, 0).reshape(-1, output_dim)
        target_output_ids = tgt[:, 1:].reshape(-1)

        loss = loss_fn(output, target_output_ids)

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate_rnn(model, valid_dataset, loss_fn, batch_size, pad_id, device) -> float:
    """Evaluates a RNN-based Seq2Seq model.

    Args:
        model: RNN-based Seq2Seq model.
        valid_dataset: Validation dataset.
        loss_fn: Loss function.
        batch_size: Batch size.
        pad_id: Padding token id.
        device: Device to use.

    Returns:
        Validation loss.
    """
    model.eval()

    losses = 0

    val_iter = valid_dataset
    val_dataloader = DataLoader(
        val_iter, batch_size=batch_size, collate_fn=CollateFunctor(pad_id)
    )

    for src, _, tgt, _, _, _ in val_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        encoder_hidden, encoder_cell = model.encoder.init_hidden(src.shape[0])

        target_input_ids = tgt[:, :-1]
        output = model(
            src,
            encoder_hidden,
            encoder_cell,
            target_input_ids,
            0,
        )
        output_dim = output.shape[-1]

        output = output.transpose(1, 0).reshape(-1, output_dim)
        target_output_ids = tgt[:, 1:].reshape(-1)

        loss = loss_fn(output, target_output_ids)
        losses += loss.item()

    return losses / len(list(val_dataloader))


def training_loop_rnn(
    model: Seq2SeqModelRNN,
    optimizer,
    lr_scheduler,
    loss_fn,
    training_dataset,
    valid_dataset,
    pad_id: int,
    teacher_forcing_ratio: float,
    batch_size: int,
    device,
    n_epochs: int,
    patience: int,
    model_path: str,
) -> None:
    """Training loop.

    Args:
        model: RNN-based Seq2Seq model.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler.
        loss_fn: Loss function.
        training_dataset: Training dataset.
        valid_dataset: Validation dataset.
        pad_id: Padding token id.
        teacher_forcing_ratio: Teacher forcing ratio.
        batch_size: Batch size.
        n_epochs: Number of epochs.
        patience: Patience.
        model_path: Path to save the model.
    """
    best_valid_loss = float("inf")
    patience_counter = 0

    logger.info("***** Running training *****")
    logger.info(f"  Model = {model.__class__.__name__}")
    logger.info(f"  Num Epochs = {n_epochs}")
    logger.info(f"  Patience = {patience}")

    for epoch in tqdm(range(1, n_epochs + 1)):
        start_time = timer()
        train_loss = train_rnn_epoch(
            model,
            optimizer,
            lr_scheduler,
            loss_fn,
            training_dataset,
            pad_id,
            batch_size,
            device,
            teacher_forcing_ratio,
        )
        end_time = timer()
        val_loss = evaluate_rnn(
            model, valid_dataset, loss_fn, batch_size, pad_id, device
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

    save_rnn(model, model_path)
