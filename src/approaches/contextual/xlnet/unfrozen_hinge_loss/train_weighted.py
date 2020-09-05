import sys

sys.path.append("./")
sys.path.append("../../../../utilities")


import os

import argparse

import torch
import transformers
from transformers import AutoTokenizer, AdamW

import math

from dataloader import DataGenerator
from model import Network
from pytorchtools import EarlyStopping


from torchsummaryX import summary

from tqdm.auto import tqdm

from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import f1_score, precision_score, recall_score

import pickle

writer = SummaryWriter("transformers_only/xlnet/logs_weighted")


def to_label(labels, x):
    return [round(round(p, 1) / x) for p in labels.flatten().tolist()]


def get_weighted_weights(dataset_path):
    dataset = pickle.load(open(dataset_path, "rb"))
    count = [0 for key in range(5)]
    for dp in dataset:
        count[dp["Label"]] += 1
    maximum_frequency = max(count)
    # total_frequency = sum(count)
    weights = []
    for c in count:
        if c == 0:
            weights.append(0)
        else:
            weight = math.log(maximum_frequency / c)
            if weight == 0:
                weight = 1.0
            # weight = math.sqrt(total_frequency/c)
            weights.append(weight)
    return weights


def get_weight_tensor(labels, weights):
    weight_tensor = []
    for label in labels:
        weight_tensor.append(weights[int(label * 4)])
    return torch.tensor(weight_tensor).cuda()


def weighted_mse_loss(target, pred, weights):
    weight_tensor = get_weight_tensor(target, weights)
    return torch.sum(weight_tensor * (pred - target) ** 2) / target.shape[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_data_path",
        type=str,
        default="transformers_only/xlnet/train.pkl",
        help="Path of the pickled list containing training samples",
    )
    parser.add_argument(
        "--val_data_path",
        type=str,
        default="transformers_only/xlnet/val.pkl",
        help="Path of the pickled list containing validation samples",
    )

    parser.add_argument("--batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--tensorboard_label_number",
        type=str,
        default="",
        help="The label for the tensorboard graph",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning Rate"
    )
    parser.add_argument(
        "--patience", type=int, default=3, help="Early Stopping patience variable"
    )

    args = parser.parse_args()

    train_data_path = args.train_data_path
    val_data_path = args.val_data_path

    batch_size = args.batch_size
    epochs = args.epochs
    patience = args.patience

    tensorboard_label_number = args.tensorboard_label_number
    learning_rate = args.learning_rate
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu")
    print("Extracting data from: " + str(train_data_path))
    print("Using batch size: " + str(batch_size))
    print("Training for " + str(epochs) + " epochs")
    print("Using " + str(device))

    # define the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlnet-base-cased")
    early_stopping = EarlyStopping(
        patience=patience,
        verbose=True,
        path="transformers_only/xlnet/logs_weighted/model/best_checkpoint_"
        + tensorboard_label_number
        + ".pth",
    )

    # define the train dataloader
    train_dataset = DataGenerator(
        data_path=train_data_path,
        batch_size=batch_size,
        tokenizer=tokenizer,
        split="training",
        device=device,
    )
    validation_dataset = DataGenerator(
        data_path=val_data_path,
        batch_size=batch_size,
        tokenizer=tokenizer,
        split="test",
        device=device,
    )

    # load the model
    model = Network(model_name="xlnet-base-cased", freeze_xlnet=True)
    model.to(device)
    # display the summary of the model
    # txt = "XLNet is pretty similar to BERT in its architecture. The major difference lies in their training objective. BERT masks the data and tries to predict the masked data using a bi-directional context whereas XLNet uses permutation objective. As the name suggests, in its most naive form it generates all permutations of words in a sentence and tries to maximize the likelihood of sequence. In this tutorial, I will show how one can re-train/fine-tune XLNet’s language model from the checkpoint and then how to use the finetuned language model for sequence classification. We will finetune the model using Tensorflow. We will then convert the finetuned TensorFlow model to Pytorch Model using one of my favorite libraries named Transformers. Then we will use the Transformers library to do sequence classification. We will also compare the results with using directly pre-trained XLNet’smodel. This is a very practical intensive tutorial. I did this on Google Colab with two Jupyter notebooks. In first, I have shown how to retrain your model from the checkpoint and then converting the re-trained model to the Pytorch model using Transformers-cli. In the second notebook, I have shown how to use the pre-trained model for sequence classification. We will also compare the results between the finetuned and pre-trained model. All the code can be found on the shared Github repository below."
    # inputs = tokenizer(txt, return_tensors="pt")
    batch = DataGenerator(
        data_path=train_data_path,
        batch_size=batch_size,
        tokenizer=tokenizer,
        split="training",
        device=device,
    ).load_batch()[0]
    print(summary(model, [i.to(device) for i in batch]))

    # get weights
    training_weights = get_weighted_weights(train_data_path)
    validation_weights = training_weights
    print(
        "training weights", training_weights, "validation weights", validation_weights
    )
    # define the loss
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.MultiLabelMarginLoss()

    # define the optimiser
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "gamma", "beta"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay_rate": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    last_epoch = 1
    # train the model
    # steps = 0
    for epoch in range(last_epoch, epochs + 1):
        print("Epoch: {}".format(epoch))
        tr_loss = 0
        val_loss = 0
        pbar = tqdm(total=math.ceil(len(train_dataset) / batch_size))
        pbar.set_description("Epoch " + str(epoch))
        i = 1
        y_true = []
        y_pred = []
        # Training Loop
        model.train()
        while True:
            optimizer.zero_grad()

            batch = train_dataset.load_batch()
            if batch is None:
                break

            inputs, label_tensor = batch
            output = model(inputs)

            y_true += label_tensor
            y_pred += output

            loss = loss_fn(label_tensor.view(-1), output.view(-1), training_weights)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()

            writer.add_scalar(
                "training loss_" + tensorboard_label_number + "_" + str(learning_rate),
                tr_loss / i,
                epoch * len(train_dataset) + i,
            )

            pbar.set_postfix_str("Loss: " + str(tr_loss / i))
            pbar.update(1)
            # print("[train] loss: {}".format(tr_loss / steps))
            # steps += 1
            i += 1
        writer.add_scalar(
            "macro f1 score train_"
            + tensorboard_label_number
            + "_"
            + str(learning_rate),
            f1_score(y_true, y_pred, average="macro"),
            i * epoch,
        )
        pbar.close()
        # Validation Loop

        if not os.path.exists("transformers_only/xlnet/logs_weighted/model"):
            os.makedirs("transformers_only/xlnet/logs_weighted/model")

        torch.save(
            model.state_dict(),
            "transformers_only/xlnet/logs_weighted/model/ckpt_"
            + str(tensorboard_label_number)
            + "_"
            + str(epoch)
            + ".pth",
        )

        i = 1
        y_true = []
        y_pred = []
        out_score = []
        pbar = tqdm(total=math.ceil(len(validation_dataset) / batch_size))
        pbar.set_description("Epoch(Validation) " + str(epoch))
        model.eval()
        while True:
            with torch.no_grad():
                batch = validation_dataset.load_batch()
                if batch is None:
                    break
                inputs, label_tensor = batch

                output = model(inputs)
                y_true += label_tensor
                y_pred += output
                out_score += [p for p in output.flatten().tolist()]
                loss = loss_fn(
                    label_tensor.view(-1), output.view(-1), validation_weights
                )
                val_loss += loss.item()
                writer.add_scalar(
                    "validation loss_"
                    + tensorboard_label_number
                    + "_"
                    + str(learning_rate),
                    val_loss / i,
                    epoch * len(validation_dataset) + i,
                )
                pbar.set_postfix_str("Val Loss: " + str(val_loss / i))
                pbar.update(1)
                i += 1

        pbar.close()

        writer.add_scalar(
            "macro f1 score val_" + tensorboard_label_number + "_" + str(learning_rate),
            f1_score(y_true, y_pred, average="macro"),
            i * epoch,
        )

        early_stopping((val_loss / (i - 1)), model)
        if early_stopping.early_stop:
            print("Early stopping!!!")
            break

        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        macrof1 = f1_score(y_true, y_pred, average="macro")

        with open(
            "transformers_only/xlnet/output_scores_weighted_tuning"
            + str(learning_rate)
            + ".txt",
            "a",
        ) as f:
            y_true_string = ",".join(str(i) for i in y_true)
            y_pred_string = ",".join(str(i) for i in y_pred)
            out_score_string = ",".join(str(i) for i in out_score)

            f.write(
                y_true_string
                + "\t"
                + y_pred_string
                + "\t"
                + out_score_string
                + "\t"
                + str(precision)
                + "\t"
                + str(recall)
                + "\t"
                + str(macrof1)
                + "\n"
            )


if __name__ == "__main__":
    main()
