import sys

sys.path.append("./")

import torch

from transformers import XLNetModel, XLNetConfig
from transformers.modeling_utils import SequenceSummary


def freeze_xlnet_fn(
    model, freeze_layers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], freeze_embeddings=True
):

    if freeze_embeddings:
        for param in list(model.word_embedding.parameters()):
            param.requires_grad = False
        print("Froze Embedding Layer")

    if len(freeze_layers) != 0:
        layer_indices = freeze_layers
        for layer_idx in layer_indices:
            for param in list(model.layer[layer_idx].parameters()):
                param.requires_grad = False
            print("Froze Layer: ", layer_idx)
            # print(model.bert.encoder.layer[layer_idx])

    return model


class Network(torch.nn.Module):
    def __init__(
        self, model_name="xlnet-base-cased", freeze_xlnet=True, freeze_layers=[]
    ):
        super(Network, self).__init__()
        self.model_name = model_name

        self.config = XLNetConfig.from_pretrained(self.model_name)
        self.xlnet = XLNetModel.from_pretrained(self.model_name, config=self.config)

        # self.sequence_summary = SequenceSummary(self.config)
        if freeze_xlnet:
            self.xlnet = freeze_xlnet_fn(self.xlnet)

        self.fc0 = torch.nn.Linear(768, 768)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(768, 1)

    def forward(self, inputs):  # attention mask, position of [mask] token
        hidden = self.xlnet(**inputs)[0]
        output = hidden[:, -1]
        output = self.fc0(output)
        output = self.relu(output)
        output = self.fc1(output)
        output = torch.sigmoid(output)
        return output
