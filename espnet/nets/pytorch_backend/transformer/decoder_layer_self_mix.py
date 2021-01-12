#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder self-attention layer definition."""

import torch
from torch import nn

from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm


class DecoderLayer(nn.Module):
    """Single decoder layer module.

    :param int size: input dim
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        self_attn: self attention module
    :param espnet.nets.pytorch_backend.transformer.attention.MultiHeadedAttention
        src_attn: source attention module
    :param espnet.nets.pytorch_backend.transformer.positionwise_feed_forward.
        PositionwiseFeedForward feed_forward: feed forward layer module
    :param float dropout_rate: dropout rate
    :param bool normalize_before: whether to use layer_norm before the first block
    :param bool concat_after: whether to concat attention layer's input and output
        if True, additional linear will be applied.
        i.e. x -> x + linear(concat(x, att(x)))
        if False, no additional linear will be applied. i.e. x -> x + att(x)

    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward_1,#fix
        feed_forward_2,#fix
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward_1 = feed_forward_1#fix
        self.feed_forward_2 = feed_forward_2#fix
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)

    def forward(self, tgt, tgt_mask, memory, memory_mask, cache=None):
        """Compute decoded features.

        Args:
            tgt (torch.Tensor):
                decoded previous target features (batch, max_time_out, size)
            tgt_mask (torch.Tensor): mask for x (batch, max_time_out, max_time_out)
            memory (torch.Tensor): encoded source features (batch, max_time_in, size)
            memory_mask (torch.Tensor): mask for memory (batch, 1, max_time_in)
            cache (torch.Tensor): cached output (batch, max_time_out-1, size)

        """
        length = memory.shape[1]
        input_1 = torch.cat((memory, tgt), dim=1)
        input_1 = self.norm1(input_1)
        memory_norm = input_1[:,:length,:]
        tgt_norm = input_1[:,length:,:]
        # two norm
        # memory_norm = self.norm1(memory)
        # tgt_norm = self.norm1(tgt)

        residual_1 = memory_norm
        residual_2 = tgt_norm

        if cache is not None:
            # print('cache is not None')
            assert cache.shape == (
                tgt.shape[0],
                tgt.shape[1] - 1,
                self.size,
            ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
            src_query = tgt_norm[:, -1:, :]
            residual_2 = residual_2[:, -1:, :]
            tgt_q_mask = None
            if tgt_mask is not None:
                tgt_q_mask = tgt_mask[:, -1:, :]
        else:
            src_query = tgt_norm
            tgt_q_mask = tgt_mask

        if memory_mask is None:#for decoding
            memory_mask = torch.ones(1, 1, memory_norm.shape[1], dtype=torch.bool)
            temp_mask = memory_mask.repeat(1,tgt_q_mask.shape[1],1)
            src_mask = torch.cat((temp_mask, tgt_q_mask), dim=-1)
        else:
            temp_mask = memory_mask.repeat(1,tgt_q_mask.shape[1],1)
            src_mask = torch.cat((temp_mask, tgt_q_mask), dim=-1)

        src_key = torch.cat((memory_norm, tgt_norm), dim=-2)
        src_value = torch.cat((memory_norm, tgt_norm), dim=-2)

        x1 = residual_1 + self.dropout(self.self_attn(memory_norm, memory_norm, memory_norm, memory_mask))
        x2 = residual_2 + self.dropout(self.src_attn(src_query, src_key, src_value, src_mask))

        length = x1.shape[1]
        input_2 = torch.cat((x1, x2), dim=1)
        input_2 = self.norm2(input_2)
        x1 = input_2[:,:length,:]
        x2 = input_2[:,length:,:]
        # two norm
        # x1 = self.norm2(x1)
        # x2 = self.norm2(x2)

        residual_1 = x1
        residual_2 = x2
        x1 = residual_1 + self.dropout(self.feed_forward_1(x1))
        x2 = residual_2 + self.dropout(self.feed_forward_2(x2))
        if cache is not None:#decoder新加的
            x2 = torch.cat([cache, x2], dim=1)
 
        return x2, tgt_mask, x1, memory_mask
