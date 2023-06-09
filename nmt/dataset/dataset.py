import torch
import torch.nn.functional as F
from smart_open import open
from tokenizers import Tokenizer
from torch.nn.utils.rnn import pad_sequence


class TranslationDataset(torch.utils.data.Dataset):
    def __init__(
        self, input_path: str, tokenizer_path: str, direction="fr->en", bpe_dropout=0.0
    ) -> None:
        """Translation dataset.

        Args:
            input_path: Path to the input data.
            tokenizer_path: Path to the tokenizer.
            direction: Translation direction. Defaults to "fr->en".
            bpe_dropout: BPE dropout probability. Defaults to 0.0.
        """
        self.tokenizer = Tokenizer.from_file(
            tokenizer_path
        )  # we load a new instance here to allow parallelized data loading
        self.tokenizer.model.dropout = bpe_dropout

        self.sentences = [
            [sentence_variant for sentence_variant in line.strip().split("\t")]
            for line in open(input_path, "r")
        ]
        if direction == "en->fr":
            self.sentences = [(en, fr) for fr, en in self.sentences]

    def __getitem__(self, index: int):
        source_str, target_str = self.sentences[index]

        source_ids = self.tokenizer.encode(source_str).ids
        target_ids = self.tokenizer.encode(target_str).ids

        return (
            torch.tensor(source_ids),
            torch.tensor(target_ids),
            source_str,
            target_str,
        )

    def __len__(self):
        return len(self.sentences)


class TokenSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: TranslationDataset,
        max_tokens: int,
        shuffle=False,
        drop_last=False,
    ) -> None:
        """Sampler that samples batches based on the number of tokens.

        Args:
            dataset: A translation dataset.
            max_tokens: Maximum number of tokens per batch.
            shuffle: Whether to shuffle the data.
            drop_last: Whether to drop the last batch if it is smaller than max_tokens.
        """
        if shuffle:
            self.sampler = torch.utils.data.RandomSampler(dataset)
        else:
            self.sampler = torch.utils.data.SequentialSampler(dataset)

        self.dataset = dataset
        self.max_tokens = max_tokens
        self.drop_last = drop_last

        total_len = sum(sample[1].size(0) for sample in self.dataset)
        self.approximate_len = total_len // self.max_tokens

    def __iter__(self):
        batch, n_tokens = [], 0
        for sample_i in self.sampler:
            _, target, _, _ = self.dataset[sample_i]

            if n_tokens + target.size(0) > self.max_tokens:
                assert len(batch) > 0
                yield batch
                batch, n_tokens = [], 0

            batch.append(sample_i)
            n_tokens += target.size(0)

        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        return (
            self.approximate_len
        )  # it would be too expensive to calculate the real length, this is actually a lower bound


class CollateFunctor:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, sentences: list):
        source_ids, target_ids, source_str, target_str = zip(*sentences)
        source_ids, source_mask = self.collate_sentences(source_ids)
        target_ids, target_mask = self.collate_sentences(target_ids)
        return source_ids, source_mask, target_ids, target_mask, source_str, target_str

    def collate_sentences(self, sentences: list):
        lengths = [sentence.size(0) for sentence in sentences]
        max_length = max(lengths)

        subword_ids = torch.stack(
            [
                F.pad(sentence, (0, max_length - length), value=self.pad_id)
                for length, sentence in zip(lengths, sentences)
            ]
        )
        attention_mask = subword_ids == self.pad_id

        return subword_ids, attention_mask


def collate_fn(batch, pad_id: int):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample, _, _ in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=pad_id)
    tgt_batch = pad_sequence(tgt_batch, padding_value=pad_id)
    return src_batch, tgt_batch
