"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn


class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, hidden_size, drop_prob=0.):
        super(BiDAF, self).__init__()
        self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob)

    def forward(self, cw_idxs, qw_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
        q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class CharBiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, n_chars, embed_size, max_word_len, hidden_size, drop_prob=0.):
        super(CharBiDAF, self).__init__()
        self.hidden_size = hidden_size
        self.charEmb = layers.CharEmbedding(n_chars=n_chars, embed_size=embed_size, max_word_len=max_word_len,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.wordEmb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFSelfAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=12 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob,
                                      multiplier=12)

    def forward(self, cc_idxs, qc_idxs, cw_idxs, qw_idxs):
        # if len(cc_idxs.shape) > 2 and len(qc_idxs.shape) > 2:
        #     c_mask = torch.zeros_like(torch.sum(cc_idxs, dim=2)) != torch.sum(cc_idxs, dim=2) # Mask if all chars are 0, and is thus a padded word
        #     q_mask = torch.zeros_like(torch.sum(qc_idxs, dim=2)) != torch.sum(qc_idxs, dim=2) # Mask if all chars are 0, and is thus a padded word
        # else:
        #     c_mask = torch.zeros_like(cc_idxs) != cc_idxs
        #     q_mask = torch.zeros_like(qc_idxs) != qc_idxs

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        cc_emb = self.charEmb(cc_idxs)  # (batch_size, c_len, hidden_size)
        qc_emb = self.charEmb(qc_idxs)  # (batch_size, q_len, hidden_size)

        cw_emb = self.wordEmb(cw_idxs)  # (batch_size, c_len, hidden_size)
        qw_emb = self.wordEmb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_emb = torch.cat([cc_emb, cw_emb], dim=-1) 
        q_emb = torch.cat([qc_emb, qw_emb], dim=-1)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 12 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out

class SelfAttCharBiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.

    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).

    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.

    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, word_vectors, n_chars, embed_size, max_word_len, hidden_size, drop_prob=0.):
        super(SelfAttCharBiDAF, self).__init__()
        self.hidden_size = hidden_size
        self.charEmb = layers.CharEmbedding(n_chars=n_chars, embed_size=embed_size, max_word_len=max_word_len,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.wordEmb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

        self.enc = layers.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

        self.self_att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                 drop_prob=drop_prob)

        self.mod = layers.RNNEncoder(input_size=14 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=2,
                                     drop_prob=drop_prob)

        self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob, multiplier=14)

    def forward(self, cc_idxs, qc_idxs, cw_idxs, qw_idxs):
        # if len(cc_idxs.shape) > 2 and len(qc_idxs.shape) > 2:
        #     c_mask = torch.zeros_like(torch.sum(cc_idxs, dim=2)) != torch.sum(cc_idxs, dim=2) # Mask if all chars are 0, and is thus a padded word
        #     q_mask = torch.zeros_like(torch.sum(qc_idxs, dim=2)) != torch.sum(qc_idxs, dim=2) # Mask if all chars are 0, and is thus a padded word
        # else:
        #     c_mask = torch.zeros_like(cc_idxs) != cc_idxs
        #     q_mask = torch.zeros_like(qc_idxs) != qc_idxs

        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs

        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        cc_emb = self.charEmb(cc_idxs)  # (batch_size, c_len, hidden_size)
        qc_emb = self.charEmb(qc_idxs)  # (batch_size, q_len, hidden_size)

        cw_emb = self.wordEmb(cw_idxs)  # (batch_size, c_len, hidden_size)
        qw_emb = self.wordEmb(qw_idxs)  # (batch_size, q_len, hidden_size)

        c_emb = torch.cat([cc_emb, cw_emb], dim=-1)
        q_emb = torch.cat([qc_emb, qw_emb], dim=-1)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 12 * hidden_size)

        c_att = att[:, :, 2*self.hidden_size:4*self.hidden_size] # (batch_size, c_len, 2 * hidden_size)

        self_att = self.self_att(c_att, c_att, c_mask, c_mask)

        att = torch.cat([att, self_att[:, :, 2*self.hidden_size:]], dim=-1) # (batch_size, c_len, 14 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        out = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
