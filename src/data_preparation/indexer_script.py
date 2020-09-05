import os
import json
from intervals import extract_podcasts
from intervals import create_intervals
from preprocess import to_lower
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--path",
    type=str,
    default="/content/drive/My Drive/Spotify",
    help="Path to Spotify folder",
)
args = parser.parse_args()
path = args.path
counter = 0
chapters = os.listdir(path + "/dataset/spotify-podcasts-2020/podcasts-transcripts")
preprocessed = []
total = 0


for chapter in range(len(chapters) - 1):

    print("CHAPTER", chapter)
    pages = os.listdir(
        path + "/dataset/spotify-podcasts-2020/podcasts-transcripts/" + str(chapter)
    )
    for page in pages:
        # podcasts from an entire page
        try:
            podcasts, file_names = extract_podcasts(chapter, page, path)
            for file_num, podcast in enumerate(podcasts):
                interval = create_intervals(podcast)
                new_data = [
                    {
                        "id": "doc_"
                        + str(counter + doc_num + 1)
                        + "_"
                        + file_names[file_num][:-5],
                        "contents": to_lower(doc),
                    }
                    for doc_num, doc in enumerate(interval)
                ]
                preprocessed += new_data
                counter = counter + len(interval) + 1
        except Exception as e:
            pass
    with open(path + "/document" + str(chapter) + ".json", "a") as f:
        f.write("[")
    for content in preprocessed:
        with open(path + "/document" + str(chapter) + ".json", "a") as f:
            f.write(json.dumps(content) + ",")

    with open(path + "/document" + str(chapter) + ".json", "rb+") as filehandle:
        filehandle.seek(-1, os.SEEK_END)
        filehandle.truncate()
    with open(path + "/document" + str(chapter) + ".json", "a") as f:
        f.write("]")
    del preprocessed
    preprocessed = []
