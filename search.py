import logging
import torch
import torch.nn as nn
import torch.nn.functional as F


class GreedySearch:
    def __init__(self, model, tokenizer, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.sos_id = tokenizer.token_to_id("[SOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.pad_id = tokenizer.token_to_id("[PAD]")

    @torch.no_grad()
    def __call__(self, source, source_mask):
        source_encoding = self.model.encode_source(source, source_mask)
        target = torch.full(
            [source_encoding.size(0), 1], fill_value=self.sos_id, device=source.device
        )
        stop = torch.zeros(target.size(0), dtype=torch.bool, device=target.device)

        for _ in range(self.max_length):
            prediction = self.model.decode_step(source_encoding, source_mask, target)
            prediction = torch.where(stop, self.pad_id, prediction.argmax(-1))
            stop |= prediction == self.eos_id

            target = torch.cat([target, prediction.unsqueeze(1)], dim=1)

            if stop.all():
                break

        sentences = self.tokenizer.decode_batch(
            target.tolist(), skip_special_tokens=True
        )
        return sentences

    @torch.no_grad()
    def rnn_call(self, source, source_hidden, source_cell):
        source_encoding, hidden, cell = self.model.encode_source(
            source, source_hidden, source_cell
        )

        target = torch.full(
            [source_encoding.size(0)], fill_value=self.sos_id, device=source.device
        )
        stop = torch.zeros(target.size(0), dtype=torch.bool, device=target.device)
        outputs = torch.full(
            [self.max_length, source_encoding.size(0)],
            fill_value=self.sos_id,
            device=source.device,
        )

        for i in range(self.max_length):
            prediction, hidden, cell = self.model.decode_step(hidden, cell, outputs[i])
            prediction = torch.where(stop, self.pad_id, prediction.argmax(-1))
            stop |= prediction == self.eos_id

            outputs[i] = prediction

            if stop.all():
                break

        outputs = outputs.transpose(1, 0)
        sentences = self.tokenizer.decode_batch(
            outputs.tolist(), skip_special_tokens=True
        )
        return sentences


class BeamSearch:
    def __init__(self, model, tokenizer, beam_size=4, max_length=128):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_size = beam_size
        self.max_length = max_length

        self.sos_id = tokenizer.token_to_id("[SOS]")
        self.eos_id = tokenizer.token_to_id("[EOS]")
        self.vocab_size = tokenizer.get_vocab_size()

    @torch.no_grad()
    def __call__(self, source, source_mask):
        batch_size = source.size(0)
        source_encoding = self.model.encode_source(source, source_mask)

        candidates = [[] for _ in range(batch_size)]

        target = torch.full(
            [batch_size, 1], fill_value=self.sos_id, device=source.device
        )
        prediction = self.model.decode_step(
            source_encoding, source_mask, target
        ).squeeze(0)
        prediction = F.log_softmax(prediction, dim=-1)
        prediction = torch.topk(
            prediction, self.beam_size, dim=-1
        )  # shape: [batch, beam]

        source_encoding = source_encoding.repeat_interleave(self.beam_size, dim=0)
        source_mask = source_mask.repeat_interleave(self.beam_size, dim=0)
        target = torch.cat(
            [
                target.repeat_interleave(self.beam_size, dim=0),
                prediction.indices.flatten().unsqueeze(1),
            ],
            dim=1,
        )
        logp = prediction.values  # shape: [batch, beam]

        for length in range(self.max_length):
            prediction = self.model.decode_step(
                source_encoding, source_mask, target
            )  # shape: [B, V]
            prediction = prediction.view(batch_size, self.beam_size, -1)
            prediction = F.log_softmax(prediction, dim=-1)  # shape: [batch, beam, V]
            prediction = logp.unsqueeze(-1) + prediction  # shape: [batch, beam, V]
            prediction = prediction.flatten(1, 2)  # shape: [batch, beam x V]
            prediction = torch.topk(
                prediction, 2 * self.beam_size, dim=1, sorted=True
            )  # shape: [batch, beam]

            target = target.cpu()
            next_subword = (
                prediction.indices % self.vocab_size
            ).tolist()  # shape: [batch, beam]
            previous_batch = (
                prediction.indices // self.vocab_size
            ).tolist()  # shape: [batch, beam]
            logp = prediction.values.tolist()  # shape: [batch, beam]
            next_target, next_logps = [], []

            for batch in range(batch_size):
                for subword, beam, score in zip(
                    next_subword[batch], previous_batch[batch], logp[batch]
                ):
                    if subword == self.eos_id:
                        if len(candidates[batch]) < self.beam_size:
                            candidates[batch].append(
                                (
                                    target[batch * self.beam_size + beam, :],
                                    score / (length + 1),
                                )
                            )
                    else:
                        next_target.append(
                            torch.cat(
                                [
                                    target[batch * self.beam_size + beam, :],
                                    torch.tensor([subword]),
                                ],
                                dim=-1,
                            )
                        )
                        next_logps.append(score)

                        if len(next_target) % self.beam_size == 0:
                            break

            if all(len(candidate) >= self.beam_size for candidate in candidates):
                break

            target = torch.stack(next_target, dim=0).to(
                source_encoding.device
            )  # shape: [batch x beam, length]
            logp = torch.tensor(next_logps, device=target.device).view(
                batch_size, -1
            )  # shape: [batch, beam]

        best_targets = []
        for batch in range(batch_size):
            if len(candidates[batch]) == 0:
                best_targets.append(next_target[batch * self.beam_size].tolist())
            else:
                best_targets.append(
                    sorted(candidates[batch], key=lambda x: x[1], reverse=True)[0][
                        0
                    ].tolist()
                )

        sentences = self.tokenizer.decode_batch(best_targets, skip_special_tokens=True)
        return sentences
