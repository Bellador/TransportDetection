import os
import json
from pathlib import Path
from queryAPI import get_results, get_videos_info
from extractTimestamps import tempony_and_weather_extraction

def ytapi_main(OUTPUT_PATH, query='Paris walk', NUMBER_OF_VIDEOS='all', PROCESS_VIDEOS_WITH_TIMESTAMPS=False, PROCESS_SPECIFIC_IDS=[]):
    print('-' * 30)
    print(f'[*] Starting YoutubeAPI query: {query}')
    api_key = open(r"./YoutubeAPI-Query/api_token.txt", 'r').readline()
    # create storage dir if needed
    STORAGE_PATH = os.path.join(OUTPUT_PATH, 'ytapi_data/')
    Path(STORAGE_PATH).mkdir(parents=True, exist_ok=True)
    # query Youtube API V3 to retrieve videoIds for given query
    get_results(query, api_key, STORAGE_PATH, NUMBER_OF_VIDEOS=NUMBER_OF_VIDEOS, PROCESS_SPECIFIC_IDS=PROCESS_SPECIFIC_IDS)
    # alter query to be used as savestring of outputfiles
    query = query.replace(" ", "_")
    # acquire more metadata for the previously acquired videoIds
    get_videos_info(query, api_key, STORAGE_PATH, NUMBER_OF_VIDEOS=NUMBER_OF_VIDEOS)
    # use pattern detection to extract video timestamps, dates and times from the textual video description
    videoIDs_with_timestamps = tempony_and_weather_extraction(query, STORAGE_PATH)
    #GEOPARSING WILL BE DONE LATER ON TO DIRECTLY MERGE WITH OCR DETECTED LOCATIONS!
    # # find matches of location names of the extracted text descriptions in a gazetteer of the given location (e.g. Paris)
    # geoparsing(query, STORAGE_PATH)
    ytquery_filepath_json = os.path.join(STORAGE_PATH, f'{query}_videoInfoExtracted.json')
    # get all the videoIds to process
    with open(ytquery_filepath_json, 'rt', encoding='utf-8') as f:
        data = json.load(f)
    if PROCESS_VIDEOS_WITH_TIMESTAMPS:
        print(f'[!] PROCESS_VIDEOS_WITH_TIMESTAMPS = True; download and process only videos with timestamps!')
        videoIDs_to_process = videoIDs_with_timestamps
    else:
        videoIDs_to_process = list(data.keys())
    return videoIDs_to_process, ytquery_filepath_json


if __name__ == '__main__':
    query = 'Paris walk'
    ytapi_main(query=query)