# freeze bert layers

from transformers import BertForMaskedLM

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
print(len(model.bert.encoder.layer))  # 12

freeze_embeddings = True
if freeze_embeddings:
    for param in list(model.bert.embeddings.parameters()):
        param.requires_grad = False
    print("Froze Embedding Layer")


freeze_layers = "0,1,2,3,4,5,6,7,8"
if freeze_layers != "":
    layer_indices = [int(x) for x in freeze_layers.split(",")]
    for layer_idx in layer_indices:
        for param in list(model.bert.encoder.layer[layer_idx].parameters()):
            param.requires_grad = False
        print("Froze Layer: ", layer_idx)
        # print(model.bert.encoder.layer[layer_idx])

print(model)
