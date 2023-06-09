"""Evaluation of RNN-based Seq2Seq model."""
import argparse
import logging
from nmt.dataset.dataset import TranslationDataset
from nmt.models.rnn import Seq2SeqModelRNN
import confuse
from nmt.utils import count_parameters
import torch
from torch import Tensor
import evaluate
from timeit import default_timer as timer
import pandas as pd

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_decode(
    model: Seq2SeqModelRNN,
    src,
    max_length: int,
    bos_idx: int,
    eos_idx: int,
    pad_id: int,
) -> Tensor:
    """Generates translation using greedy decoding.

    Args:
        model: RNN-based Seq2Seq model.
        src: Source sentence.
        max_len: Maximum length of the generated translation.
        eos_indx: End of sentence token index.
        bos_idx: Beginning of sentence token index.
        pad_id: Padding token index.

    Returns:
        Generated translation.
    """
    src = src.to(DEVICE)
    src = src.transpose(1, 0)

    encoder_hidden, encoder_cell = model.encoder.init_hidden(1)

    source_encoding, hidden, cell = model.encode_source(
        src, encoder_hidden, encoder_cell
    )

    target = torch.full(
        [source_encoding.size(0)], fill_value=bos_idx, device=src.device
    )
    stop = torch.zeros(target.size(0), dtype=torch.bool, device=target.device)
    outputs = torch.full(
        [max_length, source_encoding.size(0)],
        fill_value=bos_idx,
        device=src.device,
    )

    for i in range(max_length):
        prediction, hidden, cell = model.decode_step(hidden, cell, outputs[i])
        prediction = torch.where(stop, pad_id, prediction.argmax(-1))
        stop |= prediction == eos_idx

        outputs[i] = prediction

        if stop.all():
            break

    outputs = outputs.transpose(1, 0)
    return outputs


def translate(
    model: Seq2SeqModelRNN,
    src_sentence: str,
    tokenizer,
    bos_idx: int,
    eos_idx: int,
    pad_id: int,
) -> Tensor:
    """Translates a sentence using a RNN-based Seq2Seq model.

    Args:
        model: RNN-based Seq2Seq model.
        src_sentence: Source sentence.
        tokenizer: Tokenizer.
        bos_idx: Beginning of sentence token index.
        eos_idx: End of sentence token index.
        pad_id: Padding token index.

    Returns:
        Generated translation.
    """
    model.eval()
    src = torch.tensor(tokenizer.encode(src_sentence).ids).view(-1, 1)
    num_tokens = src.shape[0]
    tgt_tokens = greedy_decode(
        model,
        src,
        max_length=num_tokens + 5,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        pad_id=pad_id,
    ).flatten()
    return tokenizer.decode(tgt_tokens.tolist(), skip_special_tokens=True)


def parse_args() -> argparse.Namespace:
    """Defines and parses command line arguments.

    Returns:
        A namespace object containing the arguments.
    """
    parser = argparse.ArgumentParser(prog="transformer_evaluation.py")
    parser.add_argument(
        "-c",
        "--config-file",
        help=(
            "Path to configuration file to overwrite default values. "
            "Defaults to None."
        ),
    )
    return parser.parse_args()


def main(config: confuse.Configuration):
    """Performs evaluation of Transformer-based Seq2Seq model.

    Args:
        config: Configuration.
    """
    tokenizer_path = config["tokenizer"].get(str)
    test_data_path = config["test_data"].get(str)
    logger.info(f"Loading test data from {test_data_path}")
    test_dataset = TranslationDataset(
        test_data_path, tokenizer_path, direction="fr->en", bpe_dropout=0.0
    )

    bos_idx = test_dataset.tokenizer.token_to_id("[SOS]")
    eos_idx = test_dataset.tokenizer.token_to_id("[EOS]")
    pad_idx = test_dataset.tokenizer.token_to_id("[PAD]")

    model_path = config["model"].get(str)
    logger.info(f"Loading model from {model_path}.")
    state_dict = torch.load(model_path, map_location=DEVICE)
    rnn_model = Seq2SeqModelRNN(
        state_dict["vocab_size"],
        state_dict["hidden_size"],
        state_dict["num_layers"],
        state_dict["encoder_dropout"],
        state_dict["encoder_dropout"],
        DEVICE,
    )
    rnn_model.load_state_dict(state_dict["model_state_dict"])
    rnn_model.to(DEVICE)
    logger.debug(rnn_model)
    logger.debug(f"Model has {count_parameters(rnn_model)} parameters.")

    meteor = evaluate.load("meteor")
    chrf = evaluate.load("chrf")
    sacrebleu = evaluate.load("sacrebleu")

    rnn_model.eval()

    targets = []
    predictions = []

    start_time = timer()
    for _, _, src_sentence, tgt_sentence in test_dataset:
        predictions.append(
            translate(
                rnn_model,
                src_sentence,
                test_dataset.tokenizer,
                bos_idx,
                eos_idx,
                pad_idx,
            )
        )
        targets.append([tgt_sentence])
    end_time = timer()
    avg_time = (end_time - start_time) / len(test_dataset)
    print(f"Avg prediction time = {avg_time:.3f}s")

    output_file = config["output_file"].get(str)
    logger.info(f"Writing predictions to {output_file}.")
    df = pd.DataFrame(zip(predictions, targets), columns=["prediction", "target"])
    df.to_csv(output_file, index=False)

    meteor_results = meteor.compute(predictions=predictions, references=targets)
    print(f"METEOR: {meteor_results}")

    chrf_results = chrf.compute(predictions=predictions, references=targets)
    print(f"CHRF: {chrf_results}")

    chrf_pp_results = chrf.compute(
        predictions=predictions, references=targets, word_order=2
    )
    print(f"CHRF++: {chrf_pp_results}")

    bleu_results = sacrebleu.compute(predictions=predictions, references=targets)
    print(f"SacreBLEU: {bleu_results}")


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Loading configuration file at {args.config_file}.")
    config = confuse.Configuration("nmt")
    config.set_file(args.config_file)

    if config["debug"].get(bool):
        logger.setLevel(logging.DEBUG)

    main(config)
