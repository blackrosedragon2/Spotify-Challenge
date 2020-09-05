from typing import Dict, Iterator, List

import torch
import torch.nn as nn
from torch.autograd import Variable
from transformers import XLNetConfig, XLNetModel

# from allennlp.modules.text_field_embedders import TextFieldEmbedder
from allennlp.modules.matrix_attention.cosine_matrix_attention import (
    CosineMatrixAttention,
)

# from allennlp.modules.matrix_attention.dot_product_matrix_attention import *
# from allennlp.modules.seq2seq_encoders import StackedSelfAttentionEncoder,MultiHeadSelfAttention
import math

torch.autograd.set_detect_anomaly(True)


def freeze_xlnet_fn(model, freeze_layers="0,1,2,3,4,5,6,7,8,9", freeze_embeddings=True):

    freeze_embeddings = True
    if freeze_embeddings:
        for param in list(model.word_embedding.parameters()):
            param.requires_grad = False
        print("Froze Embedding Layer")

    if freeze_layers != "":
        layer_indices = [int(x) for x in freeze_layers.split(",")]
        for layer_idx in layer_indices:
            for param in list(model.layer[layer_idx].parameters()):
                param.requires_grad = False
            print("Froze Layer: ", layer_idx)
            # print(model.bert.encoder.layer[layer_idx])

    return model


class Network(nn.Module):
    """
    TK is a neural IR model - a fusion between transformer contextualization & kernel-based scoring
    -> uses 1 transformer block to contextualize embeddings
    -> soft-histogram kernels to score interactions
    """

    def __init__(
        self,
        model_name="xlnet-base-cased",
        freeze_xlnet=True,
        kernels_mu=[1.0, 0.9, 0.7, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -0.7, -0.9],
        kernels_sigma=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
    ):

        super(Network, self).__init__()

        n_kernels = len(kernels_mu)

        if len(kernels_mu) != len(kernels_sigma):
            raise Exception("len(kernels_mu) != len(kernels_sigma)")

        self.model_name = model_name

        # static - kernel size & magnitude variables
        self.mu = Variable(
            torch.cuda.FloatTensor(kernels_mu), requires_grad=False
        ).view(1, 1, 1, n_kernels)
        self.sigma = Variable(
            torch.cuda.FloatTensor(kernels_sigma), requires_grad=False
        ).view(1, 1, 1, n_kernels)
        self.nn_scaler = nn.Parameter(
            torch.full([1], 0.01, dtype=torch.float32, requires_grad=True)
        )

        self.config = XLNetConfig.from_pretrained(self.model_name)
        self.xlnet = XLNetModel.from_pretrained(self.model_name, config=self.config)

        if freeze_xlnet:
            self.xlnet = freeze_xlnet_fn(self.xlnet)

        # this does not really do "attention" - just a plain cosine matrix calculation (without learnable weights)
        self.cosine_module = CosineMatrixAttention()

        # bias is set to True in original code (we found it to not help, how could it?)
        self.dense = nn.Linear(n_kernels, 1, bias=False)
        self.dense_mean = nn.Linear(n_kernels, 1, bias=False)
        self.dense_comb = nn.Linear(2, 1, bias=False)

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(
            self.dense.weight, -0.014, 0.014
        )  # inits taken from matchzoo
        torch.nn.init.uniform_(
            self.dense_mean.weight, -0.014, 0.014
        )  # inits taken from matchzoo

        # init with small weights, otherwise the dense output is way to high for the tanh -> resulting in loss == 1 all the time
        torch.nn.init.uniform_(
            self.dense.weight, -0.014, 0.014
        )  # inits taken from matchzoo
        # self.dense.bias.data.fill_(0.0)

    def forward(self, inputs, output_secondary_output: bool = False) -> torch.Tensor:
        # pylint: disable=arguments-differ

        query_embeddings = self.xlnet(**(inputs[0]))[0].permute(0, 2, 1)
        document_embeddings = self.xlnet(**(inputs[1]))[0].permute(0, 2, 1)
        # print(query_embeddings)

        # print(query_embeddings.shape)
        # print(document_embeddings.shape)

        query_pad_oov_mask = inputs[0]["attention_mask"]
        document_pad_oov_mask = inputs[1]["attention_mask"]
        # print(query_pad_oov_mask.shape)
        # print(document_pad_oov_mask.shape)

        for i in range(query_embeddings.shape[0]):
            query_embeddings[i] = query_embeddings[i] * query_pad_oov_mask[i]
            document_embeddings[i] = document_embeddings[i] * document_pad_oov_mask[i]
        # query_embeddings = query_embeddings * query_pad_oov_mask
        # document_embeddings = document_embeddings * document_pad_oov_mask
        # print(query_embeddings.shape)
        # print(document_embeddings.shape)

        #
        # prepare embedding tensors & paddings masks
        # -------------------------------------------------------

        query_by_doc_mask = torch.bmm(
            query_pad_oov_mask.float().unsqueeze(-1),
            document_pad_oov_mask.float().unsqueeze(-1).transpose(-1, -2),
        )
        query_by_doc_mask_view = query_by_doc_mask.unsqueeze(-1)

        #
        # cosine matrix
        # -------------------------------------------------------

        # shape: (batch, query_max, doc_max)
        query_embeddings = query_embeddings.permute(0, 2, 1)
        document_embeddings = document_embeddings.permute(0, 2, 1)
        # print(query_embeddings.shape)
        # print(document_embeddings.shape)

        cosine_matrix = self.cosine_module.forward(
            query_embeddings, document_embeddings
        )
        # print(cosine_matrix.shape)

        cosine_matrix_masked = cosine_matrix * query_by_doc_mask
        cosine_matrix_extradim = cosine_matrix.unsqueeze(-1)

        #
        # gaussian kernels & soft-TF
        #
        # first run through kernel, then sum on doc dim then sum on query dim
        # -------------------------------------------------------

        raw_kernel_results = torch.exp(
            -torch.pow(cosine_matrix_extradim - self.mu, 2)
            / (2 * torch.pow(self.sigma, 2))
        )
        kernel_results_masked = raw_kernel_results * query_by_doc_mask_view

        doc_lengths = torch.sum(document_pad_oov_mask, 1)

        per_kernel_query = torch.sum(kernel_results_masked, 2)
        log_per_kernel_query = (
            torch.log2(torch.clamp(per_kernel_query, min=1e-10)) * self.nn_scaler
        )
        log_per_kernel_query_masked = (
            log_per_kernel_query * query_pad_oov_mask.unsqueeze(-1)
        )  # make sure we mask out padding values
        per_kernel = torch.sum(log_per_kernel_query_masked, 1)

        # per_kernel_query_mean = torch.sum(kernel_results_masked2_mean, 2)

        per_kernel_query_mean = per_kernel_query / (
            doc_lengths.view(-1, 1, 1) + 1
        )  # well, that +1 needs an explanation, sometimes training data is just broken ... (and nans all the things!)

        log_per_kernel_query_mean = per_kernel_query_mean * self.nn_scaler
        log_per_kernel_query_masked_mean = (
            log_per_kernel_query_mean * query_pad_oov_mask.unsqueeze(-1)
        )  # make sure we mask out padding values
        per_kernel_mean = torch.sum(log_per_kernel_query_masked_mean, 1)

        ##
        ## "Learning to rank" layer - connects kernels with learned weights
        ## -------------------------------------------------------

        dense_out = self.dense(per_kernel)
        dense_mean_out = self.dense_mean(per_kernel_mean)
        dense_comb_out = self.dense_comb(torch.cat([dense_out, dense_mean_out], dim=1))
        score = torch.squeeze(dense_comb_out, 1)  # torch.tanh(dense_out), 1)

        if output_secondary_output:
            query_mean_vector = query_embeddings.sum(dim=1) / query_pad_oov_mask.sum(
                dim=1
            ).unsqueeze(-1)
            return (
                score,
                {
                    "score": score,
                    "dense_out": dense_out,
                    "dense_mean_out": dense_mean_out,
                    "per_kernel": per_kernel,
                    "per_kernel_mean": per_kernel_mean,
                    "query_mean_vector": query_mean_vector,
                    "cosine_matrix_masked": cosine_matrix_masked,
                },
            )
        else:
            return torch.sigmoid(score)

    def get_param_stats(
        self,
    ):  # " b: "+str(self.dense.bias.data) +\ "b: "+str(self.dense_mean.bias.data) +
        return (
            "TK: dense w: "
            + str(self.dense.weight.data)
            + "dense_mean weight: "
            + str(self.dense_mean.weight.data)
            + "dense_comb weight: "
            + str(self.dense_comb.weight.data)
            + "scaler: "
            + str(self.nn_scaler.data)
        )

    def get_param_secondary(self):
        return {
            "dense_weight": self.dense.weight,  # "dense_bias":self.dense.bias,
            "dense_mean_weight": self.dense_mean.weight,  # "dense_mean_bias":self.dense_mean.bias,
            "dense_comb_weight": self.dense_comb.weight,
            "scaler": self.nn_scaler,
        }
