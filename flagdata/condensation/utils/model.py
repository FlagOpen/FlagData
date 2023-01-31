# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
from collections import OrderedDict
from torch import nn
from torch.nn import functional as F
import math
import numpy as np
import torch
from packaging import version
from transformers.activations import gelu
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import PreTrainedModel, apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from transformers import DistilBertConfig


def cross_entropy_with_soft_labels(logits, labels):
    return (-labels * F.log_softmax(logits, dim=-1)).sum(-1)


class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_classes, drop_p=0.1):
        super().__init__()

        # config
        self.bert_config = bert_model.config
        self.num_classes = num_classes

        # pretrained bert model
        self.bert_model = bert_model

        # pooler
        if not hasattr(bert_model, "pooler"):
            self.pooler = nn.Sequential(
                nn.Linear(bert_model.config.dim,
                          bert_model.config.dim), nn.Tanh()
            )
        else:
            self.pooler is None

        # dropout layer
        self.dropout = nn.Dropout(p=drop_p)
        # single linear layer for classification
        self.classifier = nn.Linear(self.bert_config.hidden_size, num_classes)

        # loss function
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        output_attentions=False,
    ):
        # encode input sequences with bert model
        bert_outputs = self.bert_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )

        # hidden state of [CLS] token
        if "pooler_output" in bert_outputs.keys():
            pooler_output = bert_outputs["pooler_output"]
        else:
            pooler_output = self.pooler(
                bert_outputs["last_hidden_state"][:, 0])

        # dropout
        pooler_output = self.dropout(pooler_output)

        # classifier layer
        logits = self.classifier(pooler_output)

        # calculate losses
        if labels is not None:
            if logits.shape == labels.shape:
                losses = cross_entropy_with_soft_labels(logits, labels)
            else:
                losses = self.cross_entropy(logits, labels)
            return losses, logits, bert_outputs

        return logits, bert_outputs

    def forward_with_params(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        labels=None,
        weights=None,
        output_attentions=False,
    ):
        assert weights is not None
        module_name_list = ["bert_model", "classifier"]
        if self.pooler is not None:
            module_name_list.append("pooler")
        weights_dict = {module_name: OrderedDict()
                        for module_name in module_name_list}
        for name, param in weights.items():
            module_name, param_name = name.split(".", maxsplit=1)
            weights_dict[module_name][param_name] = param

        # encode input sequences with bert model
        bert_outputs = self.bert_model.forward_with_params(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            weights=weights_dict["bert_model"],
            output_attentions=output_attentions,
        )

        # hidden state of [CLS] token
        if "pooler_output" in bert_outputs.keys():
            pooler_output = bert_outputs["pooler_output"]
        else:
            pooler_output = F.linear(
                bert_outputs["last_hidden_state"][:, 0],
                weight=weights_dict["pooler"]["0.weight"],
                bias=weights_dict["pooler"]["0.bias"],
            )
            pooler_output = self.pooler[1](pooler_output)

        # dropout
        pooler_output = self.dropout(pooler_output)

        # classifier layer
        logits = F.linear(
            pooler_output,
            weight=weights_dict["classifier"]["weight"],
            bias=weights_dict["classifier"]["bias"],
        )

        # calculate losses
        if labels is not None:
            if logits.shape == labels.shape:
                losses = cross_entropy_with_soft_labels(logits, labels)
            else:
                losses = self.cross_entropy(logits, labels)
            return losses, logits, bert_outputs

        return logits, bert_outputs

    def reset_additional_parameters(self):
        if self.pooler is not None:
            self.pooler[0].reset_parameters()
        self.classifier.reset_parameters()


DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "distilbert-base-uncased",
    "distilbert-base-uncased-distilled-squad",
    "distilbert-base-cased",
    "distilbert-base-cased-distilled-squad",
    "distilbert-base-german-cased",
    "distilbert-base-multilingual-cased",
    "distilbert-base-uncased-finetuned-sst-2-english",
    # See all DistilBERT models at https://huggingface.co/models?filter=distilbert
]


# UTILS AND BUILDING BLOCKS OF THE ARCHITECTURE #


def create_sinusoidal_embeddings(n_pos, dim, out):
    if is_deepspeed_zero3_enabled():
        from transformers import deepspeed

        with deepspeed.zero.GatheredParameters(out, modifier_rank=0):
            if torch.distributed.get_rank() == 0:
                _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)
    else:
        _create_sinusoidal_embeddings(n_pos=n_pos, dim=dim, out=out)


def _create_sinusoidal_embeddings(n_pos, dim, out):
    position_enc = np.array(
        [
            [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
            for pos in range(n_pos)
        ]
    )
    out.requires_grad = False
    out[:, 0::2] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
    out[:, 1::2] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
    out.detach_()


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.dim, padding_idx=config.pad_token_id
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.dim
        )
        if config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=config.max_position_embeddings,
                dim=config.dim,
                out=self.position_embeddings.weight,
            )

        self.LayerNorm = nn.LayerNorm(config.dim, eps=1e-12)
        self.dropout = nn.Dropout(config.dropout)
        if version.parse(torch.__version__) > version.parse("1.6.0"):
            self.register_buffer(
                "position_ids",
                torch.arange(config.max_position_embeddings).expand((1, -1)),
                persistent=False,
            )

    def forward(self, input_ids):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        seq_length = input_ids.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # isues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(
                input_ids
            )  # (bs, max_seq_length)

        word_embeddings = self.word_embeddings(
            input_ids)  # (bs, max_seq_length, dim)
        position_embeddings = self.position_embeddings(
            position_ids
        )  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + \
            position_embeddings  # (bs, max_seq_length, dim)
        embeddings = self.LayerNorm(embeddings)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings

    def forward_with_params(self, input_ids, weights=None):
        """
        Parameters:
            input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

        Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
        embeddings)
        """
        assert weights is not None
        module_name_list = ["word_embeddings",
                            "position_embeddings", "LayerNorm"]
        weights_dict = {module_name: OrderedDict()
                        for module_name in module_name_list}
        for name, param in weights.items():
            module_name, param_name = name.split(".", maxsplit=1)
            weights_dict[module_name][param_name] = param

        seq_length = input_ids.size(1)

        # Setting the position-ids to the registered buffer in constructor, it helps
        # when tracing the model without passing position-ids, solves
        # issues similar to issue #5664
        if hasattr(self, "position_ids"):
            position_ids = self.position_ids[:, :seq_length]
        else:
            position_ids = torch.arange(
                seq_length, dtype=torch.long, device=input_ids.device
            )  # (max_seq_length)
            position_ids = position_ids.unsqueeze(0).expand_as(
                input_ids
            )  # (bs, max_seq_length)

        word_embeddings = F.embedding(
            input_ids, weights_dict["word_embeddings"]["weight"]
        )  # (bs, max_seq_length, dim)
        position_embeddings = F.embedding(
            position_ids, weights_dict["position_embeddings"]["weight"]
        )  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + \
            position_embeddings  # (bs, max_seq_length, dim)
        embeddings = F.layer_norm(
            embeddings,
            (embeddings.size(-1),),
            weight=weights_dict["LayerNorm"]["weight"],
            bias=weights_dict["LayerNorm"]["bias"],
            eps=1e-12,
        )  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = nn.Linear(
            in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

    def prune_heads(self, heads):
        attention_head_size = self.dim // self.n_heads
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.n_heads, attention_head_size, self.pruned_heads
        )
        # Prune linear layers
        self.q_lin = prune_linear_layer(self.q_lin, index)
        self.k_lin = prune_linear_layer(self.k_lin, index)
        self.v_lin = prune_linear_layer(self.v_lin, index)
        self.out_lin = prune_linear_layer(self.out_lin, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.dim = attention_head_size * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        query,
        key,
        value,
        mask,
        head_mask=None,
        output_attentions=False,
    ):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return (
                x.transpose(1, 2).contiguous().view(
                    bs, -1, self.n_heads * dim_per_head)
            )

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        # (bs, n_heads, q_length, dim_per_head)
        q = q / math.sqrt(dim_per_head)
        # (bs, n_heads, q_length, k_length)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))

        # (bs, n_heads, q_length, k_length)
        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        # (bs, n_heads, q_length, dim_per_head)
        context = torch.matmul(weights, v)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)

    def forward_with_params(
        self,
        query,
        key,
        value,
        mask,
        head_mask=None,
        output_attentions=False,
        weights=None,
    ):
        """
        Parameters:
            query: torch.tensor(bs, seq_length, dim)
            key: torch.tensor(bs, seq_length, dim)
            value: torch.tensor(bs, seq_length, dim)
            mask: torch.tensor(bs, seq_length)

        Returns:
            weights: torch.tensor(bs, n_heads, seq_length, seq_length) Attention weights context: torch.tensor(bs,
            seq_length, dim) Contextualized layer. Optional: only if `output_attentions=True`
        """
        assert weights is not None
        module_name_list = ["q_lin", "k_lin", "v_lin", "out_lin"]
        weights_dict = {module_name: OrderedDict()
                        for module_name in module_name_list}
        for name, param in weights.items():
            module_name, param_name = name.split(".", maxsplit=1)
            weights_dict[module_name][param_name] = param

        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, f'Dimensions do not match: {dim} input vs {self.dim} configured'
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """separate heads"""
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """group heads"""
            return (
                x.transpose(1, 2).contiguous().view(
                    bs, -1, self.n_heads * dim_per_head)
            )

        # q_lin
        q = shape(
            F.linear(
                query,
                weight=weights_dict["q_lin"]["weight"],
                bias=weights_dict["q_lin"]["bias"],
            )
        )  # (bs, n_heads, q_length, dim_per_head)
        # k_lin
        k = shape(
            F.linear(
                key,
                weight=weights_dict["k_lin"]["weight"],
                bias=weights_dict["k_lin"]["bias"],
            )
        )  # (bs, n_heads, k_length, dim_per_head)
        # v_lin
        v = shape(
            F.linear(
                value,
                weight=weights_dict["v_lin"]["weight"],
                bias=weights_dict["v_lin"]["bias"],
            )
        )  # (bs, n_heads, k_length, dim_per_head)

        # (bs, n_heads, q_length, dim_per_head)
        q = q / math.sqrt(dim_per_head)
        # (bs, n_heads, q_length, k_length)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (
            (mask == 0).view(mask_reshp).expand_as(scores)
        )  # (bs, n_heads, q_length, k_length)
        # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))

        # (bs, n_heads, q_length, k_length)
        weights = nn.Softmax(dim=-1)(scores)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        # (bs, n_heads, q_length, dim_per_head)
        context = torch.matmul(weights, v)
        context = unshape(context)  # (bs, q_length, dim)
        # out_lin
        context = F.linear(
            context,
            weight=weights_dict["out_lin"]["weight"],
            bias=weights_dict["out_lin"]["bias"],
        )  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context,)


class FFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.lin1 = nn.Linear(in_features=config.dim,
                              out_features=config.hidden_dim)
        self.lin2 = nn.Linear(
            in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in [
            "relu",
            "gelu",
        ], f"activation ({config.activation}) must be in ['relu', 'gelu']"
        self.activation = gelu if config.activation == "gelu" else nn.ReLU()

    def forward(self, input):

        return apply_chunking_to_forward(
            self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, input
        )

    def forward_with_params(self, input, weights=None):
        assert weights is not None
        module_name_list = ["lin1", "lin2"]
        weights_dict = {module_name: OrderedDict()
                        for module_name in module_name_list}
        for name, param in weights.items():
            module_name, param_name = name.split(".", maxsplit=1)
            weights_dict[module_name][param_name] = param

        def ff_chunk_with_params(input):
            x = F.linear(
                input,
                weight=weights_dict["lin1"]["weight"],
                bias=weights_dict["lin1"]["bias"],
            )
            x = self.activation(x)
            x = F.linear(
                x,
                weight=weights_dict["lin2"]["weight"],
                bias=weights_dict["lin2"]["bias"],
            )
            x = self.dropout(x)
            return x

        return apply_chunking_to_forward(
            ff_chunk_with_params, self.chunk_size_feed_forward, self.seq_len_dim, input
        )

    def ff_chunk(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = nn.LayerNorm(
            normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = nn.LayerNorm(
            normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output

    def forward_with_params(
        self, x, attn_mask=None, head_mask=None, output_attentions=False, weights=None
    ):
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim)
            attn_mask: torch.tensor(bs, seq_length)

        Returns:
            sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length) The attention weights ffn_output:
            torch.tensor(bs, seq_length, dim) The output of the transformer block contextualization.
        """
        assert weights is not None
        module_name_list = ["attention",
                            "sa_layer_norm", "ffn", "output_layer_norm"]
        weights_dict = {module_name: OrderedDict()
                        for module_name in module_name_list}
        for name, param in weights.items():
            module_name, param_name = name.split(".", maxsplit=1)
            weights_dict[module_name][param_name] = param

        # Self-Attention
        sa_output = self.attention.forward_with_params(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            weights=weights_dict["attention"],
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attentions` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        # sa_layer_norm
        sa_output = F.layer_norm(
            sa_output + x,
            (sa_output.size(-1),),
            weight=weights_dict["sa_layer_norm"]["weight"],
            bias=weights_dict["sa_layer_norm"]["bias"],
            eps=1e-12,
        )  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn.forward_with_params(
            sa_output, weights=weights_dict["ffn"]
        )  # (bs, seq_length, dim)
        ffn_output = F.layer_norm(
            ffn_output + sa_output,
            (ffn_output.size(-1),),
            weight=weights_dict["output_layer_norm"]["weight"],
            bias=weights_dict["output_layer_norm"]["bias"],
            eps=1e-12,
        )  # (bs, seq_length, dim)

        output = (ffn_output,)
        if output_attentions:
            output = (sa_weights,) + output
        return output


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.layer = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

    def forward(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(
                x=hidden_state,
                attn_mask=attn_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

    def forward_with_params(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
        weights=None,
    ):  # docstyle-ignore
        """
        Parameters:
            x: torch.tensor(bs, seq_length, dim) Input sequence embedded.
            attn_mask: torch.tensor(bs, seq_length) Attention mask on the sequence.

        Returns:
            hidden_state: torch.tensor(bs, seq_length, dim) Sequence of hidden states in the last (top)
            layer all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
                Tuple of length n_layers with the hidden states from each layer.
                Optional: only if output_hidden_states=True
            all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
                Tuple of length n_layers with the attention weights from each layer
                Optional: only if output_attentions=True
        """
        assert weights is not None
        layer_weights = OrderedDict({i: OrderedDict()
                                    for i in range(self.n_layers)})
        for name, param in weights.items():
            module_name, layer_id, param_name = name.split(".", maxsplit=2)
            assert module_name == "layer"
            layer_weights[int(layer_id)][param_name] = param

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module.forward_with_params(
                x=hidden_state,
                attn_mask=attn_mask,
                head_mask=head_mask[i],
                output_attentions=output_attentions,
                weights=layer_weights[i],
            )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_state, all_hidden_states, all_attentions]
                if v is not None
            )
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

# INTERFACE FOR ENCODER AND TASK SPECIFIC MODEL #


class DistilBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DistilBertConfig
    load_tf_weights = None
    base_model_prefix = "distilbert"

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class DistilBertModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = Embeddings(config)  # Embeddings
        self.transformer = Transformer(config)  # Encoder

        self.init_weights()

    def get_position_embeddings(self) -> nn.Embedding:
        """
        Returns the position embeddings
        """
        return self.embeddings.position_embeddings

    def resize_position_embeddings(self, new_num_position_embeddings: int):
        """
        Resizes position embeddings of the model if :obj:`new_num_position_embeddings !=
        config.max_position_embeddings`.

        Arguments:
            new_num_position_embeddings (:obj:`int`):
                The number of new position embedding matrix. If position embeddings are learned, increasing the size
                will add newly initialized vectors at the end, whereas reducing the size will remove vectors from the
                end. If position embeddings are not learned (*e.g.* sinusoidal position embeddings), increasing the
                size will add correct vectors at the end following the position encoding algorithm, whereas reducing
                the size will remove vectors from the end.
        """
        num_position_embeds_diff = (
            new_num_position_embeddings - self.config.max_position_embeddings
        )

        # no resizing needs to be done if the length stays the same
        if num_position_embeds_diff == 0:
            return

        self.config.max_position_embeddings = new_num_position_embeddings

        old_position_embeddings_weight = (
            self.embeddings.position_embeddings.weight.clone()
        )

        self.embeddings.position_embeddings = nn.Embedding(
            self.config.max_position_embeddings, self.config.dim
        )

        if self.config.sinusoidal_pos_embds:
            create_sinusoidal_embeddings(
                n_pos=self.config.max_position_embeddings,
                dim=self.config.dim,
                out=self.position_embeddings.weight,
            )
        else:
            with torch.no_grad():
                if num_position_embeds_diff > 0:
                    self.embeddings.position_embeddings.weight[
                        :-num_position_embeds_diff
                    ] = nn.Parameter(old_position_embeddings_weight)
                else:
                    self.embeddings.position_embeddings.weight = nn.Parameter(
                        old_position_embeddings_weight[:num_position_embeds_diff]
                    )
        # move position_embeddings to correct device
        self.embeddings.position_embeddings.to(self.device)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.embeddings.word_embeddings = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.transformer.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)  # (bs, seq_length, dim)
        return self.transformer(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def forward_with_params(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        weights=None,
    ):

        assert weights is not None
        module_name_list = ["embeddings", "transformer"]
        weights_dict = {module_name: OrderedDict()
                        for module_name in module_name_list}
        for name, param in weights.items():
            module_name, param_name = name.split(".", maxsplit=1)
            weights_dict[module_name][param_name] = param

        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(
                input_shape, device=device)  # (bs, seq_length)

        # Prepare head mask if needed
        head_mask = self.get_head_mask(
            head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings.forward_with_params(
                input_ids, weights=weights_dict["embeddings"]
            )  # (bs, seq_length, dim)
        return self.transformer.forward_with_params(
            x=inputs_embeds,
            attn_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            weights=weights_dict["transformer"],
        )
