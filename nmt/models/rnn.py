"""Seq2Seq model with RNN for machine translation."""
import torch
import torch.nn as nn
from typing import Tuple


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size: int, num_layers: int, device, dropout: int = 0.1):
        """Initializes a LSTM Encoder.

        Args:
            hidden_size: Size of the hidden layer.
            num_layers: Number of layers in the LSTM.
            device: Device to run the encoder on.
            dropout: Dropout rate. Defaults to 0.1.
        """
        super(LSTMEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(
            hidden_size,
            hidden_size,
            num_layers,
            dropout=dropout,
            batch_first=True,
            device=device,
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
            self.num_layers, batch_size, self.hidden_size, device=self.device
        ), torch.zeros(
            self.num_layers, batch_size, self.hidden_size, device=self.device
        )


class LSTMDecoder(nn.Module):
    def __init__(
        self,
        output_size: int,
        hidden_size: int,
        num_layers: int,
        device,
        dropout: int = 0.1,
    ) -> None:
        """Initializes a LSTM Decoder.

        Args:
            output_size: Size of the output vocabulary.
            hidden_size: Size of the hidden layer.
            num_layers: Number of layers in the LSTM.
            device: Device to run the decoder on.
            dropout: Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers, dropout=dropout, device=device
        )

        self.fc_out = nn.Linear(hidden_size, output_size, device=device)

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
        device,
        name: str = "Seq2Seq",
    ) -> None:
        """Initialize a Seq2Seq model without attention.

        Args:
            vocab_size: Size of the vocabulary.
            hidden_size: Size of the hidden layers.
            num_layers: Number of layers in the LSTM.
            encoder_dropout: Dropout rate for the encoder.
            decoder_dropout: Dropout rate for the decoder.
            device: Device to run the model on.
            name: Name of the model. Defaults to "Seq2Seq".
        """
        super().__init__()
        self._name = name
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.device = device

        self.embedding = nn.Embedding(vocab_size, hidden_size, device=device)
        self.encoder = LSTMEncoder(hidden_size, num_layers, device, encoder_dropout)
        self.decoder = LSTMDecoder(
            vocab_size, hidden_size, num_layers, device, decoder_dropout
        )

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
        source = source.to(self.device)
        source = self.embedding(source)
        encoder_hidden = encoder_hidden.to(self.device)
        encoder_cell = encoder_cell.to(self.device)
        target = target.to(self.device)

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
            target_length, batch_size, target_vocab_size, device=self.device
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
