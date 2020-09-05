import json

with open("topics.json") as f:
    topics = json.loads(f.read())
# print(topics)

with open("labels.txt") as f:
    labels = f.read().split("\n")
label_tuples = []
for label in labels:
    out = label.split("\t")
    label_tuples.append(out[:3] + [int(out[3]) / 4])
# print(label_tuples)
# print(label_tuples[-1])
label_batch = []
for label_tuple in label_tuples:
    query = (
        topics[int(label_tuple[0]) - 1]["query"]
        + topics[int(label_tuple[0]) - 1]["description"]
    )
    episode = label_tuple[1].split(":")[2]
    timestamp = label_tuple[2]
    label_batch.append([query, episode, timestamp])
print(label_batch)
# query+desc,interval , label
