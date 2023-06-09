import math
from timeit import default_timer as timer
from typing import Tuple

import evaluate
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_LANGUAGE = "fr"
TGT_LANGUAGE = "en"

NUM_EPOCHS = 25
MAX_LENGTH = 100
BATCH_SIZE = 128


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


training_data_path = "data/train_government.txt"
valid_data_path = "data/valid_government.txt"
tokenizer_path = "data/gpu_tokenizer_20k.json"
bpe_dropout = 0.0

training_dataset = TranslationDataset(
    training_data_path,
    tokenizer_path,
    direction="fr->en",
    bpe_dropout=bpe_dropout,
)

valid_dataset = TranslationDataset(
    valid_data_path,
    tokenizer_path,
    direction="fr->en",
    bpe_dropout=bpe_dropout,
)

PAD_IDX = training_dataset.tokenizer.token_to_id("[PAD]")
BOS_IDX = training_dataset.tokenizer.token_to_id("[SOS]")
EOS_IDX = training_dataset.tokenizer.token_to_id("[EOS]")


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, dropout: int = 0.1):
        """Initializes a LSTM Encoder.

        Args:
            hidden_size: Size of the hidden layer.
            num_layers: Number of layers in the LSTM.
            dropout: Dropout rate. Defaults to 0.1.
        """
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            device=DEVICE,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, source: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propogates source through the network.

        Args:
            source: Embedded source sentence.
            hidden: Hidden state of the encoder.
            cell: Cell state of the encoder.

        Returns:
            Context vector.
        """
        embedded = self.dropout(source)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        return output, hidden, cell

    def init_hidden(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initializes the hidden and cell states of the encoder."""
        return torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=DEVICE
        ), torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)


class LSTMDecoder(nn.Module):
    def __init__(
        self, output_size: int, hidden_size: int, num_layers: int, dropout: int = 0.1
    ) -> None:
        """Initializes a LSTM Decoder.

        Args:
            output_size: Size of the output vocabulary.
            hidden_size: Size of the hidden layer.
            num_layers: Number of layers in the LSTM.
            dropout: Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, dropout=dropout, device=DEVICE
        )

        self.fc_out = nn.Linear(hidden_size, output_size, device=DEVICE)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, target: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propogates target through the network.

        Args:
            target: Embedded target sentence.
            hidden: Hidden state of the encoder.
            cell: Cell state of the encoder.

        Returns:
            Decoded output.
        """
        embedded = self.dropout(target)
        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        predictions = self.fc_out(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2SeqModelRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        encoder_dropout: float,
        decoder_dropout: float,
        name: str = "Seq2Seq",
    ) -> None:
        """Initialize a Seq2Seq model without attention.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Size of the hidden layers.
            num_layers: Number of layers in the LSTM.
            encoder_dropout: Dropout rate for the encoder.
            decoder_dropout: Dropout rate for the decoder.
            name: Name of the model. Defaults to "Seq2Seq".
        """
        super().__init__()
        self._name = name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout

        self.embedding = nn.Embedding(vocab_size, hidden_size, device=DEVICE)
        self.encoder = LSTMEncoder(hidden_size, num_layers, encoder_dropout)
        self.decoder = LSTMDecoder(vocab_size, hidden_size, num_layers, decoder_dropout)

    @property
    def name(self) -> str:
        """Name of the model."""
        return self._name

    def forward(
        self,
        source: torch.Tensor,
        encoder_hidden: torch.Tensor,
        encoder_cell: torch.Tensor,
        target: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            source: Tokenized source sentence. Shape [B, L].
            encoder_hidden: Hidden state of the encoder. Shape [N, B, H].
            encoder_cell: Cell state of the encoder. Shape [N, B, H].
            target: Tokenized target sentence. Shape [B, L].
            teacher_forcing_ratio: Ratio of teacher forcing. Defaults to 0.5.

        Returns:
            Tensor with predictions for each token.
        """
        source = source.to(DEVICE)
        source = self.embedding(source)
        encoder_hidden = encoder_hidden.to(DEVICE)
        encoder_cell = encoder_cell.to(DEVICE)
        target = target.to(DEVICE)

        batch_size = target.shape[0]
        target_length = target.shape[1]
        target_vocab_size = self.decoder.fc_out.out_features

        _, encoder_hidden, encoder_cell = self.encoder(
            source, encoder_hidden, encoder_cell
        )

        input = target[:, 0]
        decoder_hidden = encoder_hidden
        decoder_cell = encoder_cell
        outputs = torch.zeros(
            target_length, batch_size, target_vocab_size, device=DEVICE
        )
        for t in range(1, target_length):
            input = self.embedding(input.unsqueeze(0))
            output, decoder_hidden, decoder_cell = self.decoder(
                input, decoder_hidden, decoder_cell
            )
            outputs[t] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = target[:, t] if teacher_force else top1
        return outputs

    def encode_source(
        self, source: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encodes the source sentence.

        Args:
            source: Tokenized source sentence.
            hidden: Hidden state of the encoder.
            cell: Cell state of the encoder.

        Returns:
            Encoder outputs, hidden and cell states.
        """
        source = self.embedding(source)
        return self.encoder(source, hidden, cell)

    def decode_step(
        self,
        source_hidden: torch.Tensor,
        source_cell: torch.Tensor,
        target_prefix: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decodes a single step.

        Args:
            source_hidden: Hidden state of the encoder.
            source_cell: Cell state of the encoder.
            target_prefix: Tokenized target sentence prefix.

        Returns:
            Decoded output, hidden and cell state.
        """
        input = self.embedding(target_prefix.unsqueeze(0))
        return self.decoder(input, source_hidden, source_cell)


SRC_VOCAB_SIZE = len(training_dataset.tokenizer.get_vocab())
TGT_VOCAB_SIZE = len(training_dataset.tokenizer.get_vocab())
DROPOUT = 0.1
HIDDEN_SIZE = 256
NUM_LAYER = 3

rnn_model = Seq2SeqModelRNN(
    SRC_VOCAB_SIZE,
    HIDDEN_SIZE,
    NUM_LAYER,
    DROPOUT,
    DROPOUT,
    "Seq2SeqRNN",
).to(DEVICE)


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


def train_epoch(
    model,
    optimizer,
    lr_scheduler,
    criterion,
    training_dataset,
    teacher_forcing_ratio=0.5,
):
    model.train()

    losses = 0
    train_iter = training_dataset
    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, collate_fn=CollateFunctor(PAD_IDX)
    )

    progress_bar = tqdm(range(len(train_dataloader)))
    for src, _, tgt, _, _, _ in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

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

        loss = criterion(output, target_output_ids)

        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)

        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate_model(model, valid_dataset, criterion):
    model.eval()

    losses = 0

    val_iter = valid_dataset
    val_dataloader = DataLoader(
        val_iter, batch_size=BATCH_SIZE, collate_fn=CollateFunctor(PAD_IDX)
    )

    for src, _, tgt, _, _, _ in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

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

        loss = criterion(output, target_output_ids)
        losses += loss.item()

    return losses / len(list(val_dataloader))


best_valid_loss = float("inf")
patience_counter = 0
learning_rate = 0.00001

optimizer = torch.optim.Adam(
    rnn_model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9
)
lr_scheduler = cosine_schedule_with_warmup(optimizer, 100, NUM_EPOCHS * 391, 0.01)

criterion = nn.CrossEntropyLoss()

for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    start_time = timer()
    train_loss = train_epoch(
        rnn_model, optimizer, lr_scheduler, criterion, training_dataset
    )
    end_time = timer()
    val_loss = evaluate_model(rnn_model, valid_dataset, criterion)
    print(
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
        print("Early stopping.")
        break


def greedy_decode(model, src, max_length=MAX_LENGTH):
    src = src.to(DEVICE)
    src = src.transpose(1, 0)

    encoder_hidden, encoder_cell = model.encoder.init_hidden(1)

    source_encoding, hidden, cell = model.encode_source(
        src, encoder_hidden, encoder_cell
    )

    target = torch.full(
        [source_encoding.size(0)], fill_value=BOS_IDX, device=src.device
    )
    stop = torch.zeros(target.size(0), dtype=torch.bool, device=target.device)
    outputs = torch.full(
        [max_length, source_encoding.size(0)],
        fill_value=BOS_IDX,
        device=src.device,
    )

    for i in range(max_length):
        prediction, hidden, cell = model.decode_step(hidden, cell, outputs[i])
        prediction = torch.where(stop, PAD_IDX, prediction.argmax(-1))
        stop |= prediction == EOS_IDX

        outputs[i] = prediction

        if stop.all():
            break

    outputs = outputs.transpose(1, 0)
    return outputs


def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = torch.tensor(training_dataset.tokenizer.encode(src_sentence).ids).view(-1, 1)
    num_tokens = src.shape[0]
    tgt_tokens = greedy_decode(model, src, max_length=num_tokens + 5).flatten()
    return training_dataset.tokenizer.decode(
        tgt_tokens.tolist(), skip_special_tokens=True
    )


model_path = "data/models/rnn_gov_30.pt"
torch.save(
    {
        "vocab_size": SRC_VOCAB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYER,
        "decoder_dropout": DROPOUT,
        "encoder_dropout": DROPOUT,
        "model_state_dict": rnn_model.state_dict(),
    },
    model_path,
)


# model_path = "data/models/rnn_gov_jupyter_30.pt"
# state_dict = torch.load(model_path)
# transformer = Seq2SeqModelRNN(state_dict["vocab_size"], state_dict["hidden_size"], state_dict["num_layers"], state_dict["encoder_dropout"], state_dict["encoder_dropout"])
# transformer.load_state_dict(state_dict["model_state_dict"])
# transformer.to(DEVICE)

sentence_sample = [
    (
        "Je présume qu’il n’est pas constamment en liberté sur la lande.",
        "I suppose that it does not always run loose upon the moor.",
    ),
    (
        "Je ne sais pas ce que vous entendez par là.",
        "I don’t know what you mean by that.",
    ),
    ("– Qui était l’homme ?", '"Who was the man?"'),
    ("– Quoi, alors ?", '"What then?"'),
]

for src_sentence, tgt_sentence in sentence_sample:
    print(f"Source: {src_sentence}")
    print(f"Target: {tgt_sentence}")
    print(f"Predicted: {translate(rnn_model, src_sentence)}\n")

book_data_path = "data/test_book.txt"

book_dataset = TranslationDataset(
    book_data_path,
    tokenizer_path,
    direction="fr->en",
    bpe_dropout=bpe_dropout,
)

rnn_model.eval()

targets = []
predictions = []

for _, _, src_sentence, tgt_sentence in book_dataset:
    predictions.append(translate(rnn_model, src_sentence))
    targets.append([tgt_sentence])

df = pd.DataFrame(zip(predictions, targets), columns=["prediction", "target"])
df.to_csv("data/predictions/rnn_book_gov.csv", index=False)


meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
sacrebleu = evaluate.load("sacrebleu")


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
