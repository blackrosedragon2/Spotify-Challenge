from pyserini.search import SimpleSearcher
from pyserini import index

parser = argparse.ArgumentParser()
parser.add_argument(
    "--spotify_path",
    type=str,
    default="/content/drive/My Drive/Spotify",
    help="Path to Spotify folder",
)
parser.add_argument(
    "--output_path", type=str, default="/content", help="Path to output directory",
)

args = parser.parse_args()
path = args.path
ouput_path = args.ouput_path

# Reading Topics file and saving query names to a list --------------------
topicsToIndex = []
with open(path + "/dataset/spotify-podcasts-2020/training_dataset/topics.json") as f:
    allTopics = json.loads(f.read())
for topic in allTopics:
    topicsToIndex.append(topic["query"])

# Reading training data file and saving data to a dictionary (for each topic) ------------

documentsToScore = {str(i + 1): [] for i in range(len(topicsToIndex))}
# each dict obj , topicNum:[documentID1,documentID2....]
all_documents = []
with open(path + "/dataset/spotify-podcasts-2020/training_dataset/labels.txt") as f:
    labels = f.read().split("\n")
for i, label in enumerate(labels):
    details = label.split("\t")
    documentId = (
        "doc_" + str(int(float(details[2]) / 60) + 1) + "_" + details[1].split(":")[-1]
    )
    documentsToScore[str(details[0])].append(documentId)
    all_documents.append(documentId)

# BM25 + RM3 -----------------------------------------
searcher = SimpleSearcher(path + "/jsons/index")
index_utils = index.IndexReader(path + "/jsons/index")

searcher.set_bm25(0.9, 0.4)
searcher.set_rm3(10, 10, 0.5)

bm25_output = {str(i + 1): [] for i in range(len(topicsToIndex))}
# each dict obj , topicNum:[documentID1,documentID2....]

episodes_covered = []
for i, topic in enumerate(topicsToIndex):
    hits = searcher.search(topic, k=1000)
    for j in range(len(hits)):
        if hits[j].docid in documentsToScore[str(i + 1)]:
            bm25_output[str(i + 1)].append([hits[j].docid, hits[j].score])
            episodes_covered.append(hits[j].docid)


for episode in all_documents:
    if episode not in episodes_covered:
        score = index_utils.compute_query_document_score(episode, topic)
        bm25_output[output].append([episode, score])


with open(ouput_path, "w") as f:
    f.write(json.dumps(bm25_output))
