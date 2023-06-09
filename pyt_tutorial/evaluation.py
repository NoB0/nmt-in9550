from dataset.dataset import TranslationDataset
import evaluate
import torch
import pandas as pd
import torch.nn as nn
import math
from torch import Tensor
from torch.nn import Transformer
from training.utils import count_parameters
from timeit import default_timer as timer

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SRC_LANGUAGE = "fr"
TGT_LANGUAGE = "en"

book_data_path = "data/diy.txt"
tokenizer_path = "data/gpu_tokenizer_20k.json"
bpe_dropout = 0.0

valid_dataset = TranslationDataset(
    book_data_path,
    tokenizer_path,
    direction="fr->en",
    bpe_dropout=bpe_dropout,
)

meteor = evaluate.load("meteor")
chrf = evaluate.load("chrf")
sacrebleu = evaluate.load("sacrebleu")

BOS_IDX = valid_dataset.tokenizer.token_to_id("[SOS]")
EOS_IDX = valid_dataset.tokenizer.token_to_id("[EOS]")


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
    src = torch.tensor(valid_dataset.tokenizer.encode(src_sentence).ids).view(-1, 1)
    # print(src)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model, src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX
    ).flatten()
    return valid_dataset.tokenizer.decode(tgt_tokens.tolist(), skip_special_tokens=True)


model_path = "/home/stud/nbernard/bhome/repositories/nmt-fr-en/data/models/transformer_combined_2head_256h_256emb.pt"

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

print(transformer)
print(
    state_dict["num_encoder_layers"],
    state_dict["num_decoder_layers"],
    state_dict["emb_size"],
    state_dict["num_heads"],
    state_dict["vocab_size"],
    state_dict["vocab_size"],
    state_dict["hidden_size"],
)
print(count_parameters(transformer))

transformer.eval()

targets = []
predictions = []

start_time = timer()
for _, _, src_sentence, tgt_sentence in valid_dataset:
    predictions.append(translate(transformer, src_sentence))
    targets.append([tgt_sentence])
end_time = timer()
avg_time = (end_time - start_time) / len(valid_dataset)
print(f"Avg prediction time = {avg_time:.3f}s")
df = pd.DataFrame(zip(predictions, targets), columns=["prediction", "target"])
df.to_csv("data/predictions/transformer_diy_combined_256emb_256h_2h.csv", index=False)

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
