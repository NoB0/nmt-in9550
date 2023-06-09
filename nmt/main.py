"""Main command line application for training a model."""
import logging
from nmt.dataset.dataset import TranslationDataset
from nmt.models.rnn import Seq2SeqModelRNN
from nmt.models.transformer import Seq2SeqTransformer
from nmt.training.train_rnn import training_loop_rnn
from nmt.training.train_transformer import training_loop_transformer
from nmt.utils import cosine_schedule_with_warmup
import torch
import argparse
import confuse
import torch.nn as nn

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = "config/transformer_default_config.yaml"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.autograd.set_detect_anomaly(True)


def main(config: confuse.Configuration) -> None:
    """Trains a MT model.

    Args:
        args: Command line arguments.
    """
    training_data_path = config["training_data"].get(str)
    valid_data_path = config["validation_data"].get(str)
    tokenizer_path = config["tokenizer"].get(str)

    logger.info(f"Loading training data from {training_data_path}")
    training_dataset = TranslationDataset(
        training_data_path,
        tokenizer_path,
        direction="fr->en",
        bpe_dropout=config["bpe_dropout"].get(float),
    )

    logger.info(f"Loading validation data from {valid_data_path}")
    valid_dataset = TranslationDataset(
        valid_data_path,
        tokenizer_path,
        direction="fr->en",
        bpe_dropout=config["bpe_dropout"].get(float),
    )

    logger.info(f"Training data size: {len(training_dataset)}")
    logger.info(f"Validation data size: {len(valid_dataset)}")

    pad_id = training_dataset.tokenizer.token_to_id("[PAD]")

    logger.info(f"Creating model")
    model_type = config["model_type"].get(str)
    vocab_size = training_dataset.tokenizer.get_vocab_size()
    model = _get_model(config, model_type, vocab_size)
    logger.info(model)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

    batch_size = config["batch_size"].get(int)
    n_epochs = config["epochs"].get(int)
    patience = config["patience"].get(int)
    model_path = config["model_path"].get(str)

    if model_type == "transformer":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
        )
        training_loop_transformer(
            model,
            optimizer,
            loss_fn,
            training_dataset,
            valid_dataset,
            DEVICE,
            batch_size,
            n_epochs,
            patience,
            pad_id,
            model_path,
        )
    elif model_type == "rnn":
        teacher_forcing_ratio = config["teacher_forcing_ratio"].get(float)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.00001, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = cosine_schedule_with_warmup(optimizer, 100, n_epochs * 391, 0.01)
        training_loop_rnn(
            model,
            optimizer,
            lr_scheduler,
            loss_fn,
            training_dataset,
            valid_dataset,
            pad_id,
            teacher_forcing_ratio,
            batch_size,
            DEVICE,
            n_epochs,
            patience,
            model_path,
        )


def _get_model(
    config: confuse.Configuration, model_type: str, vocab_size: int = None
) -> torch.nn.Module:
    """Createa a translation model from a configuration.

    Args:
        config: A configuration object.
        model_type: The type of model to create.
        vocab_size: The size of the vocabulary. Defaults to None.

    Returns:
        A translation model.
    """
    if model_type == "rnn":
        return Seq2SeqModelRNN(
            vocab_size,
            config["model_config"]["hidden_size"].get(int),
            config["model_config"]["encoder_num_layers"].get(int),
            config["model_config"]["encoder_dropout"].get(float),
            config["model_config"]["decoder_dropout"].get(float),
            DEVICE,
            config["model_name"].get(str),
        ).to(DEVICE)
    elif model_type == "transformer":
        model = Seq2SeqTransformer(
            config["model_config"]["encoder_num_layers"].get(int),
            config["model_config"]["decoder_num_layers"].get(int),
            config["model_config"]["embedding_size"].get(int),
            config["model_config"]["num_heads"].get(int),
            vocab_size,
            vocab_size,
            config["model_config"]["hidden_size"].get(int),
        )
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model.to(DEVICE)
    else:
        raise ValueError(f"Unknown model type {model_type}")


def parse_args() -> argparse.Namespace:
    """Defines and parses command line arguments.

    Returns:
        A namespace object containing the arguments.
    """
    parser = argparse.ArgumentParser(prog="main.py")
    parser.add_argument(
        "-c",
        "--config-file",
        help=(
            "Path to configuration file to overwrite default values. "
            "Defaults to None."
        ),
    )
    parser.add_argument(
        "--training_data",
        type=str,
        help="Path to the training data.",
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        help="Path to the validation data.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to the tokenizer.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Output path to save trained model.",
    )

    parser.add_argument("--bpe_dropout", type=float, default=0.0, help="BPE dropout.")

    training_group = parser.add_argument_group("Training", "Training configuration.")
    training_group.add_argument(
        "--patience", type=int, help="Patience for early stopping."
    )
    training_group.add_argument(
        "--epochs", type=int, help="Number of epochs to train for."
    )
    training_group.add_argument(
        "--batch_size",
        type=int,
        help="Batch size.",
    )
    training_group.add_argument(
        "--teacher_forcing_ratio",
        type=float,
        help="Teacher forcing ratio.",
    )

    model_group = parser.add_argument_group("Model", "Model configuration.")
    model_group.add_argument("--model_type", type=str, help="Model type.")
    model_group.add_argument("--model_name", type=str, help="Model name.")
    model_group.add_argument("--hidden_size", type=int, help="Hidden size.")
    model_group.add_argument(
        "--encoder_num_layers", type=int, help="Number of encoder layers."
    )
    model_group.add_argument(
        "--decoder_num_layers", type=int, help="Number of decoder layers."
    )
    model_group.add_argument(
        "--encoder_dropout",
        type=float,
        help="Encoder dropout. ",
    )
    model_group.add_argument(
        "--decoder_dropout",
        type=float,
        help="Decoder dropout.",
    )
    model_group.add_argument(
        "--num_heads",
        type=int,
        help="Number of attention heads.",
    )
    model_group.add_argument(
        "--embedding_size",
        type=int,
        help="Embedding size.",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace) -> confuse.Configuration:
    """Loads config from config file and command line parameters.

    Loads default values from `config/default_config.yaml`. Values are
    then updated with any value specified in the command line arguments.

    Args:
        args: Arguments parsed with argparse.
    """
    # Load default config
    config = confuse.Configuration("nmt")
    config.set_file(DEFAULT_CONFIG_PATH)

    # Load additional config (update defaults).
    if args.config_file:
        config.set_file(args.config_file)

    # Update config from command line arguments
    config.set_args(args, dots=True)

    # Save run config to metadata file
    output_name = config["model_name"].get()
    with open(f"data/runs/{output_name}.meta.yaml", "w") as f:
        f.write(config.dump())
    logger.info(f"Saved run config to data/runs/{output_name}.meta.yaml")
    return config


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args)
    main(config)
