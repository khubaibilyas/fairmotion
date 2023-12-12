# Copyright (c) Facebook, Inc. and its affiliates.

import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch.nn.init import xavier_uniform_

from fairmotion.models import decoders


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.5, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class Custom(nn.Module):
    """RNN model for sequence prediction. The model uses a single RNN module to
    take an input pose, and generates a pose prediction for the next time step.

    Attributes:
        input_dim: Size of input vector for each time step
        hidden_dim: RNN hidden size
        num_layers: Number of layers of RNN cells
        dropout: Probability of an element to be zeroed
        device: Device on which to run the RNN module
    """

    def __init__(self, ntoken, ninp, num_heads, hidden_dim, num_layers, dropout=0.5):
        super(Custom, self).__init__()
        # self.pos_encoder = PositionalEncoding(ninp, dropout)
        # encoder_layers = TransformerEncoderLayer(
        #     ninp, num_heads, hidden_dim, dropout
        # )
        # self.transformer_encoder = TransformerEncoder(
        #     encoder_layers, num_layers
        # )
        # Use Linear instead of Embedding for continuous valued input
        # self.encoder = nn.Linear(ntoken, ninp)
        # self.ninp = ninp
        # self.decoder = decoders.LSTMDecoder(
        #     input_dim=ntoken, hidden_dim=hidden_dim, output_dim=ntoken,
        # )
        # self.num_layers = num_layers

        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTM(
            input_size=ntoken,
            hidden_size=hidden_dim,
            num_layers=num_layers,
        )
        self.project_to_output = nn.Linear(hidden_dim, hidden_dim)

        # decoder
        decoder_layer = TransformerDecoderLayer(ninp, num_heads, hidden_dim, dropout)
        self.transformer_decoder = TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=LayerNorm(ninp),
        )
        # Use Linear instead of Embedding for continuous valued input
        self.encoder = nn.Linear(ntoken, ninp)
        self.project = nn.Linear(ninp, ntoken)
        self.ninp = ninp
        self.num_layers = num_layers

        self.init_weights()

    def init_weights(self):
        """Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
        self.lstm.init_weights()

    def run_lstm(
        self,
        inputs,
        outputs,
        max_len=None,
        state=None,
        teacher_forcing_ratio=0,
    ):
        output = torch.zeros(inputs[0].shape).unsqueeze(0)
        if max_len is None:
            max_len = inputs.shape[0]
        else:
            teacher_forcing_ratio = 0

        for t in range(max_len):
            if t >= inputs.shape[0]:
                input = output.unsqueeze(0)
            else:
                input = inputs[t].unsqueeze(0)
            teacher_force = random.random() < teacher_forcing_ratio
            if t > 0 and not teacher_force:
                input = output.unsqueeze(0)
            output, state = self.lstm(input, state)
            output = self.project_to_output(output)
            output = output.squeeze(0)
            outputs[t] = output
        return outputs, state

    def forward(self, src, tgt, max_len=None, teacher_forcing_ratio=0.5):
        """
        Inputs:
            src, tgt: Tensors of shape (batch_size, seq_len, input_dim)
            max_len: Maximum length of sequence to be generated during
                inference. Set None during training.
            teacher_forcing_ratio: Probability of feeding gold target pose as
                decoder input instead of predicted pose from previous time step
        """
        # convert src, tgt to (seq_len, batch_size, input_dim) format
        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        lstm_input = self.dropout(src)
        state = None
        # Generate as many poses as in tgt during training
        max_len = tgt.shape[0] if max_len is None else max_len
        encoder_outputs = torch.zeros(src.shape).to(src.device)
        _, state = self.run_lstm(
            lstm_input,
            encoder_outputs,
            state=None,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        if self.training:
            decoder_outputs = torch.zeros(max_len - 1, src.shape[1], src.shape[2]).to(
                src.device
            )
            tgt = self.dropout(tgt)
            new_encoder_output, _ = self.run_lstm(
                tgt[:-1],
                decoder_outputs,
                state=state,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )
            # output = self.transformer_decoder(
            #     decoder_outputs, encoder_output, tgt_mask=tgt_mask,
            # )
            # output = self.project(output)
            # outputs = torch.cat((encoder_outputs, decoder_outputs))

            # Create mask for training
            tgt_mask = self._generate_square_subsequent_mask(tgt.shape[0]).to(
                device=tgt.device,
            )

            # Use last source pose as first input to decoder
            tgt = torch.cat((src[-1].unsqueeze(0), tgt[:-1]))
            pos_encoder_tgt = self.pos_encoder(self.encoder(tgt) * np.sqrt(self.ninp))
            output = self.transformer_decoder(
                pos_encoder_tgt,
                new_encoder_output,
                tgt_mask=tgt_mask,
            )
            output = self.project(output)
        else:
            del encoder_outputs
            new_encoder_output = torch.zeros(max_len, src.shape[1], src.shape[2]).to(
                src.device
            )
            inputs = lstm_input[-1].unsqueeze(0)
            new_encoder_output, _ = self.run_lstm(
                inputs,
                new_encoder_output,
                state=state,
                max_len=max_len,
                teacher_forcing_ratio=0,
            )

            ### add
            # greedy decoding
            decoder_input = torch.zeros(
                max_len,
                src.shape[1],
                src.shape[-1],
            ).type_as(src.data)
            next_pose = tgt[0].clone()

            # Create mask for greedy encoding across the decoded output
            tgt_mask = self._generate_square_subsequent_mask(max_len).to(
                device=tgt.device
            )

            for i in range(max_len):
                decoder_input[i] = next_pose
                pos_encoded_input = self.pos_encoder(
                    self.encoder(decoder_input) * np.sqrt(self.ninp)
                )
                decoder_outputs = self.transformer_decoder(
                    pos_encoded_input,
                    new_encoder_output,
                    tgt_mask=tgt_mask,
                )
                output = self.project(decoder_outputs)
                next_pose = output[i].clone()
                del output
            output = decoder_input
        return output.transpose(0, 1)
