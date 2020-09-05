import os
import json


def extract_podcasts(chapter, page, path):
    """
        Extract all json files of a given chapter and page 
        Args:
        - chapter: One of the outer folders numbered from 0 - 7
        - page: An inner folder indexed from 0-9 and A-Z (String)
        
        Returns a list of podcasts 
        Example:
            out = extract_content("0", "0")
            [{results:<podcast data 1>},{results:<podcast data 2>},...] 
    """
    os.chdir(
        path
        + "/dataset/spotify-podcasts-2020/podcasts-transcripts/"
        + str(chapter)
        + "/"
        + str(page)
    )
    podcasts = []
    folder_names = os.listdir()
    file_names = []
    for i, folder in enumerate(folder_names):
        podcast = {"results": []}
        file_names += os.listdir(folder)
        for json_file in os.listdir(folder):
            with open(folder + "/" + json_file) as f:
                # podcast["results"] = (
                #     podcast["results"] + json.loads(f.read())["results"]
                # )
                podcasts.append({"results": json.loads(f.read())["results"]})
    os.chdir(path)
    return podcasts, file_names


def create_intervals(podcast):
    """
        Given a podcast, this function will partition the content in approximate 2 minute intervals   
        Args:
        - podcast: Content for one podcast (object/dict)
        Returns a list of strings, each element representing a 2 minute interval 
        Example:
            out = extract_content("0", "0")
            intervals = create_intervals(out[0])
            [{id:doc0,content:'Welcome to Joe Girardi narrow...'}, {id:doc1,content:'really something that was spoken...'}, {id:doc2,content:'going to the mall or...'},...]
    """
    total_duration = 0
    text = ""
    podcast_list = []
    prev_text = ""
    for i in range(len(podcast["results"]) - 1):
        try:
            for words in podcast["results"][i]["alternatives"][0]["words"]:
                duration = float(words["endTime"][:-1]) - float(words["startTime"][:-1])
                total_duration += duration
                if total_duration <= 60:
                    # if duration < 60 prepare to write it in transcript with previous segment
                    text += words["word"] + " "
                else:

                    if len(prev_text) != 0:
                        podcast_list.append(prev_text + text)
                    total_duration = 0
                    prev_text = text
                    text = words["word"] + " "
        except:
            pass
    podcast_list.append(prev_text + text)
    return podcast_list
