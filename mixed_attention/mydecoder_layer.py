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
        mixed_attn,
        feed_forward_enc,
        feed_forward_dec,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
    ):
        """Construct an DecoderLayer object."""
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.mixed_attn = mixed_attn
        # feed forwar 應該會有兩個 for enc 跟 dec
        self.feed_forward_enc = feed_forward_enc
        self.feed_forward_dec = feed_forward_dec

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

        dim_enc = memory.size()[1]
        dim_dec = tgt.size()[1]
        print('dim_enc : ' , dim_enc)

        # concat encoder feature and decode feature
        concat_enc_dec = torch.cat((memory, tgt), 1)
        print(concat_enc_dec.size())
    
        
        concat_enc_dec = self.norm1(concat_enc_dec)


        # self and mixed attention part
        enc = concat_enc_dec[:, :dim_enc, :]
        dec = concat_enc_dec[:, dim_enc:, :]

        residual_enc = enc
        residual_dec = dec
        
        # enc self attention part

        # 這邊attn 的參數input 還要再詳細改 應該是只會吃到自己的東西
        enc = self.self_attn(enc, enc, enc, tgt_mask)

        # enc , dec mixed attention part
        # 參數應該會吃到enc跟dec跟 token 第幾個這樣m or 看mask可以知道
        # q,k,v,mask
        dec = self.mix_attn(enc, torch.cat((enc, dec), 1), torch.cat((enc, dec), 1), tgt_mask)
        
        

        # residual block
        enc = residual_enc + enc
        dec = residual_dec + dec
        
        concat_enc_dec = torch.cat((enc, dec),1)


        # second layer normalize
        concat_enc_dec = self.norm2(concat_enc_dec)


        
        enc = concat_enc_dec[:, :dim_enc, :]
        dec = concat_enc_dec[:, dim_enc:, :]

        residual_enc = enc
        residual_dec = dec

        # two feed forward fro encoder feature and decode feature
        enc = self.feed_forward_enc(enc)
        dec = self.feed_forward_dec(dec)

        enc = residual_enc + enc
        dec = residual_dec + dec 

        return dec, tgt_mask, enc, memory_mask


        # if cache is None:
        #     tgt_q = tgt
        #     tgt_q_mask = tgt_mask
        # else:
        #     # compute only the last frame query keeping dim: max_time_out -> 1
        #     assert cache.shape == (
        #         tgt.shape[0],
        #         tgt.shape[1] - 1,
        #         self.size,
        #     ), f"{cache.shape} == {(tgt.shape[0], tgt.shape[1] - 1, self.size)}"
        #     tgt_q = tgt[:, -1:, :]
        #     residual = residual[:, -1:, :]
        #     tgt_q_mask = None
        #     if tgt_mask is not None:
        #         tgt_q_mask = tgt_mask[:, -1:, :]

        # if self.concat_after:
        #     tgt_concat = torch.cat(
        #         (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)), dim=-1
        #     )
        #     x = residual + self.concat_linear1(tgt_concat)
        # else:
        #     x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask))
        # if not self.normalize_before:
        #     x = self.norm1(x)

        # residual = x
        # if self.normalize_before:
        #     x = self.norm2(x)
        # if self.concat_after:
        #     x_concat = torch.cat(
        #         (x, self.src_attn(x, memory, memory, memory_mask)), dim=-1
        #     )
        #     x = residual + self.concat_linear2(x_concat)
        # else:
        #     x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask))
        # if not self.normalize_before:
        #     x = self.norm2(x)

        # residual = x
        # if self.normalize_before:
        #     x = self.norm3(x)
        # x = residual + self.dropout(self.feed_forward(x))
        # if not self.normalize_before:
        #     x = self.norm3(x)

        # if cache is not None:
        #     x = torch.cat([cache, x], dim=1)

        # return x, tgt_mask, memory, memory_mask
