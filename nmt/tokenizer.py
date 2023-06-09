import argparse
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.decoders import WordPiece
from tokenizers import pre_tokenizers, Regex, normalizers


SPECIAL_TOKENS = ["[SOS]", "[EOS]", "[UNK]", "[PAD]"]
LOWERCASE = False


def define_tokenizer(vocab_size: int):
    model = BPE(unk_token="[UNK]", continuing_subword_prefix="##")
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
    )
    tokenizer = Tokenizer(model)
    tokenizer.normalizer = normalizers.Sequence(
        [
            normalizers.Replace(Regex("\t"), " "),
            normalizers.Replace(Regex(" {2,}"), " "),
            normalizers.BertNormalizer(
                lowercase=LOWERCASE, clean_text=True, strip_accents=False
            ),
        ]
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = TemplateProcessing(
        single="[SOS] $0 [EOS]",
        special_tokens=[("[SOS]", 0), ("[EOS]", 1)],
    )
    tokenizer.decoder = WordPiece(
        prefix="##", cleanup=True
    )  # we use WordPiece just because of the whitespace cleanup

    return tokenizer, trainer


def test(tokenizer, text):
    subwords = tokenizer.encode(text).tokens
    return " ".join(subwords)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, required=True)
    parser.add_argument("--input_file", type=str, default="data/train_combined.txt")
    parser.add_argument("--output_file", type=str, default="data/vocab.json")
    args = parser.parse_args()

    tokenizer, trainer = define_tokenizer(args.vocab_size)
    tokenizer.train([args.input_file], trainer)
    tokenizer.save(args.output_file)

    print("\nTesting the tokenizer...\n")
    tokenizer = Tokenizer.from_file(
        args.output_file
    )  # this is how to load the save tokenizer
    texts = [
        """One of the most impressive long term hobby projects is Robert's Rocket Project. He started building a 100 lbf liquid engine in 2001, fired a regeneratively cooled version in 2007, started building a regen 250 lbf in 2008.""",
        """Une idée soudaine me traversa l'esprit, et je pris la bougie des mains du maître d'hôtel.""",
    ]
    for text in texts:
        print(
            f"INPUT:  {text}\nTOKENS: {test(tokenizer, text)}\nDECODED: {tokenizer.decode(tokenizer.encode(text).ids)}\n",
            flush=True,
        )
