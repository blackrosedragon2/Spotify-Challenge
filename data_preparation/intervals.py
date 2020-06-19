import os


def extract_podcasts(chapter, page: str):
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
    os.chdir(dir + "/podcasts-transcripts/" + chapter + "/" + page)
    podcasts = []
    for i, folder in enumerate(os.listdir()):
        podcast = {"results": []}
        for json_file in os.listdir(folder):
            with open(folder + "/" + json_file) as f:
                podcast["results"] = (
                    podcast["results"] + json.loads(f.read())["results"]
                )
        podcasts.append(podcast)
    return podcasts


def create_intervals(podcast):
    """
        Given a podcast, this function will partition the content in approximate 2 minute intervals   
        Args:
        - podcast: Content for one podcast (object/dict)

        Returns a list of strings, each element representing a 2 minute interval 

        Example:
            out = extract_content("0", "0")
            intervals = create_intervals(out[0])
            ['Welcome to Joe Girardi narrow...', 'really something that was spoken...', 'going to the mall or...',...]

    """
    total_duration = 0
    text = ""
    podcast_list = []
    # loops over 1 podcast's transcripts
    for i in range(len(podcast["results"])):
        try:
            for words in podcast["results"][i]["alternatives"][0]["words"]:
                duration = float(words["endTime"][:-1]) - float(words["startTime"][:-1])
                total_duration += duration
                if total_duration <= 120:
                    # if duration < 120 prepare to write it in transcript
                    text += words["word"] + " "
                else:
                    podcast_list.append(text)
                    total_duration = 0
                    text = words["word"] + " "
        except Exception as e:
            # print('Exception :'+str(e))
            pass
    return podcast_list
