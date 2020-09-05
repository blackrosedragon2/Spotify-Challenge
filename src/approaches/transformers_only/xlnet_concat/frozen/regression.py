import sys

sys.path.append("./")

import torch

from transformers import XLNetModel, XLNetConfig
from transformers.modeling_utils import SequenceSummary


class Network(torch.nn.Module):
    def __init__(
        self, model_name="xlnet-base-cased", freeze_xlnet=True, freeze_layers=[]
    ):
        super(Network, self).__init__()
        self.model_name = model_name

        self.config = XLNetConfig.from_pretrained(self.model_name)
        self.config.output_hidden_states = True
        self.xlnet = XLNetModel.from_pretrained(self.model_name, config=self.config)

        # self.sequence_summary = SequenceSummary(self.config)

        if freeze_xlnet:
            for parameter in self.xlnet.parameters():
                parameter.requires_grad = False

        self.fc0 = torch.nn.Linear(768 * 3, 768)
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(768, 1)

    def forward(self, inputs):  # attention mask, position of [mask] token
        hidden = torch.cat(self.xlnet(**inputs)[1][10:], 2)
        output = hidden[:, -1]
        output = self.fc0(output)
        output = self.relu(output)
        output = self.fc1(output)
        output = torch.sigmoid(output)
        return output
