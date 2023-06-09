import math
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torchmetrics
from tokenizers import Tokenizer
from torch import Tensor
from torch.nn import Transformer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import evaluate as hf_evaluate
import pandas as pd

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_LANGUAGE = "fr"
TGT_LANGUAGE = "en"

training_data_path = "data/train_combined.txt"
valid_data_path = "data/train_combined.txt"
tokenizer_path = "data/gpu_tokenizer_20k.json"
bpe_dropout = 0.0


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


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[: token_embedding.size(0), :]
        )


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        emb_size: int,
        nhead: int,
        src_vocab_size: int,
        tgt_vocab_size: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_tok_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask
        )


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


SRC_VOCAB_SIZE = len(training_dataset.tokenizer.get_vocab())
TGT_VOCAB_SIZE = len(training_dataset.tokenizer.get_vocab())
EMB_SIZE = 256
NHEAD = 2
FFN_HID_DIM = 256
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(
    NUM_ENCODER_LAYERS,
    NUM_DECODER_LAYERS,
    EMB_SIZE,
    NHEAD,
    SRC_VOCAB_SIZE,
    TGT_VOCAB_SIZE,
    FFN_HID_DIM,
)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(
    transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample, _, _ in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = training_dataset
    train_dataloader = DataLoader(
        train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn
    )

    progress_bar = tqdm(range(len(train_dataloader)))
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

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


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = valid_dataset
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

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


NUM_EPOCHS = 25

best_valid_loss = float("inf")
patience_counter = 0

for epoch in tqdm(range(1, NUM_EPOCHS + 1)):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
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

model_path = "data/models/transformer_combined_2head_256h_256emb.pt"
torch.save(
    {
        "vocab_size": SRC_VOCAB_SIZE,
        "emb_size": EMB_SIZE,
        "hidden_size": FFN_HID_DIM,
        "num_encoder_layers": NUM_ENCODER_LAYERS,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "num_heads": NHEAD,
        "model_state_dict": transformer.state_dict(),
    },
    model_path,
)

BOS_IDX = training_dataset.tokenizer.token_to_id("[SOS]")
EOS_IDX = training_dataset.tokenizer.token_to_id("[EOS]")


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(
            DEVICE
        )
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# actual function to translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = torch.tensor(training_dataset.tokenizer.encode(src_sentence).ids).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    return training_dataset.tokenizer.decode(
        tgt_tokens.tolist(), skip_special_tokens=True
    )


# model_path = "data/models/transformer_jupyter.pt"
# state_dict = torch.load(model_path)
# transformer = Seq2SeqTransformer(state_dict["num_encoder_layers"], state_dict["num_decoder_layers"], state_dict["emb_size"], state_dict["num_heads"], state_dict["vocab_size"],state_dict["vocab_size"], state_dict["hidden_size"])
# transformer.load_state_dict(state_dict["model_state_dict"])
# transformer.to(DEVICE)


transformer.eval()

valid_predictions = []
valid_targets = []
for _, _, src_sentence, tgt_sentence in valid_dataset:
    predicted = translate(transformer, src_sentence)
    valid_predictions.append(predicted)
    valid_targets.append([tgt_sentence])

df = pd.DataFrame(
    zip(valid_predictions, valid_targets), columns=["prediction", "target"]
)
df.to_csv(
    "data/predictions/transformer_2heads_256h_256emb_valid_combined.csv", index=False
)


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
    print(f"Predicted: {translate(transformer, src_sentence)}\n")

book_data_path = "data/test_book.txt"

book_dataset = TranslationDataset(
    book_data_path,
    tokenizer_path,
    direction="fr->en",
    bpe_dropout=bpe_dropout,
)

transformer.eval()

targets = []
predictions = []

for _, _, src_sentence, tgt_sentence in book_dataset:
    predictions.append(translate(transformer, src_sentence))
    targets.append([tgt_sentence])

df = pd.DataFrame(zip(predictions, targets), columns=["prediction", "target"])
df.to_csv(
    "data/predictions/transformer_2heads_256h_256emb_book_combined.csv", index=False
)


meteor = hf_evaluate.load("meteor")
chrf = hf_evaluate.load("chrf")
sacrebleu = hf_evaluate.load("sacrebleu")


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

print("Evaluation on validation set")
meteor_results = meteor.compute(predictions=valid_predictions, references=valid_targets)
print(f"METEOR: {meteor_results}")

chrf_results = chrf.compute(predictions=valid_predictions, references=valid_targets)
print(f"CHRF: {chrf_results}")

chrf_pp_results = chrf.compute(
    predictions=valid_predictions, references=valid_targets, word_order=2
)
print(f"CHRF++: {chrf_pp_results}")

bleu_results = sacrebleu.compute(
    predictions=valid_predictions, references=valid_targets
)
print(f"SacreBLEU: {bleu_results}")
