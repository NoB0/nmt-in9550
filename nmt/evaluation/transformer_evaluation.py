"""Evaluation of Transformer-based Seq2Seq model."""
import argparse
from nmt.dataset.dataset import TranslationDataset
from nmt.models.transformer import Seq2SeqTransformer
from nmt.utils import count_parameters, generate_square_subsequent_mask
import torch
from torch import Tensor
import logging
import confuse
import evaluate
from timeit import default_timer as timer
import pandas as pd

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def greedy_decode(
    model, src, src_mask, max_len, bos_idx: int, eos_idx: int, device
) -> Tensor:
    """Generates translation using greedy decoding.

    Args:
        model: Transformer-based Seq2Seq model.
        src: Source sentence.
        src_mask: Source sentence mask.
        max_len: Maximum length of the generated translation.
        eos_indx: End of sentence token index.
        bos_idx: Beginning of sentence token index.
        device: Device to use.

    Returns:
        Generated translation.
    """
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(bos_idx).type(torch.long).to(device)
    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            device
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == eos_idx:
            break
    return ys


def translate(
    model: torch.nn.Module,
    src_sentence: str,
    tokenizer,
    bos_idx: int,
    eos_idx: int,
    device,
):
    """Translates a sentence using a Transformer-based Seq2Seq model.

    Args:
        model: Transformer-based Seq2Seq model.
        src_sentence: Source sentence.
        tokenizer: Tokenizer.
        bos_idx: Beginning of sentence token index.
        eos_idx: End of sentence token index.
        device: Device to use.

    Returns:
        Generated translation.
    """
    model.eval()
    src = torch.tensor(tokenizer.encode(src_sentence).ids).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,
        src,
        src_mask,
        max_len=num_tokens + 5,
        bos_idx=bos_idx,
        eos_idx=eos_idx,
        device=device,
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

    model_path = config["model"].get(str)
    logger.info(f"Loading model from {model_path}.")
    state_dict = torch.load(model_path, map_location=DEVICE)
    transformer = Seq2SeqTransformer(
        state_dict["num_encoder_layers"],
        state_dict["num_decoder_layers"],
        state_dict["emb_size"],
        state_dict["num_heads"],
        state_dict["vocab_size"],
        state_dict["vocab_size"],
        state_dict["hidden_size"],
    )
    transformer.load_state_dict(state_dict["model_state_dict"])
    transformer.to(DEVICE)
    logger.debug(transformer)
    logger.debug(f"Model has {count_parameters(transformer)} parameters.")

    meteor = evaluate.load("meteor")
    chrf = evaluate.load("chrf")
    sacrebleu = evaluate.load("sacrebleu")

    transformer.eval()

    targets = []
    predictions = []

    start_time = timer()
    for _, _, src_sentence, tgt_sentence in test_dataset:
        predictions.append(
            translate(
                transformer,
                src_sentence,
                test_dataset.tokenizer,
                bos_idx,
                eos_idx,
                DEVICE,
            )
        )
        targets.append([tgt_sentence])
    end_time = timer()
    avg_time = (end_time - start_time) / len(test_dataset)
    logger.info(f"Avg prediction time = {avg_time:.3f}s")

    output_file = config["output_file"].get(str)
    logger.info(f"Writing predictions to {output_file}.")
    df = pd.DataFrame(zip(predictions, targets), columns=["prediction", "target"])
    df.to_csv(output_file, index=False)

    meteor_results = meteor.compute(predictions=predictions, references=targets)
    logger.info(f"METEOR: {meteor_results}")

    chrf_results = chrf.compute(predictions=predictions, references=targets)
    logger.info(f"CHRF: {chrf_results}")

    chrf_pp_results = chrf.compute(
        predictions=predictions, references=targets, word_order=2
    )
    logger.info(f"CHRF++: {chrf_pp_results}")

    bleu_results = sacrebleu.compute(predictions=predictions, references=targets)
    logger.info(f"SacreBLEU: {bleu_results}")


if __name__ == "__main__":
    args = parse_args()

    logger.info(f"Loading configuration file at {args.config_file}.")
    config = confuse.Configuration("nmt")
    config.set_file(args.config_file)

    if config["debug"].get(bool):
        logger.setLevel(logging.DEBUG)

    main(config)
