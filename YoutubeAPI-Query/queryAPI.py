import os
import json
import requests
import urllib.parse

def get_url(url):
    headers = {'Accept': 'application/json'}
    r = requests.get(url, headers=headers)
    return r.json()


def add_to_storage(storage_, json_, processed_videoIds, potential_dubplicates):
    video_ids = []
    for item_ in json_['items']:
        # get videoId
        try:
            video_id = item_['id']['videoId']
            if video_id not in processed_videoIds:
                video_ids.append(video_id)
                processed_videoIds.append(video_id)
            else:
                potential_dubplicates += 1
        except KeyError:
            continue
    storage_['videoIds'] += video_ids


def get_results(q, api_key, STORAGE_PATH, NUMBER_OF_VIDEOS='all', PROCESS_SPECIFIC_IDS=[]):
    '''
    get all videos (+ pagination) for a given YouTube video search string
    issue with a limited amount of ~500 results per query, therefore splitting by up publishedBefore and publishedAfter
    (https://issuetracker.google.com/issues/35171641?pli=1)

    - check for duplicates at the end
    :return:
    '''
    print(f'--------------------------------------------------------------')
    print(f'[*] Fetching YT video metadata for search: {q}')
    # define output filename based on query
    output_filename = f'{q.replace(" ", "_")}_videoIds.json'
    output_filepath = os.path.join(STORAGE_PATH, output_filename)
    # check if file already exists, if yes skip API calls
    if os.path.isfile(output_filepath):
        print(f'[!] YT api results already acquired. Continue with metadata.')
        return None

    # only search for video IDs if none are specified
    if not PROCESS_SPECIFIC_IDS:
        # define storage for query and resulting videoId's
        storage_ = {'videoIds': []}
        # keep track of processed ids to avoid duplicates
        processed_videoIds = []
        potential_duplicates = 0
        # url encode query
        q = urllib.parse.quote(q)
        # split the query by year to retrieve the full amount of data
        for year in range(2016, 2023):
            date_before = f'{year}-01-01T00:00:00Z'
            date_after = f'{year-1}-01-01T00:00:00Z'
            url = f'https://youtube.googleapis.com/youtube/v3/search?maxResults=50&q={q}&publishedAfter={date_after}&publishedBefore={date_before}&key={api_key}'
            r_json = get_url(url)
            # print(json.dumps(r_json, indent=4))
            query_nr_results = len(r_json['items'])
            # add videoIds to storage
            add_to_storage(storage_, r_json, processed_videoIds, potential_duplicates)
            if NUMBER_OF_VIDEOS != 'all':
                if len(processed_videoIds) >= NUMBER_OF_VIDEOS:
                    print(f'[+] fetched max. number of videos: {NUMBER_OF_VIDEOS}')
                    break
            # perform pagination
            # define counters
            page = 0
            print(f'[+] page {page+1}, {year-1} - {year}, query: {query_nr_results}; total: {len(processed_videoIds)}')
            while True:
                # only return the top 4 pages, including the 200 most relevant results
                if page == 4:
                    break
                try:
                    next_page_token = r_json['nextPageToken']
                    page += 1
                    url = f'https://youtube.googleapis.com/youtube/v3/search?maxResults=50&q={q}&pageToken={next_page_token}&publishedAfter={date_after}&publishedBefore={date_before}&key={api_key}'
                    r_json = get_url(url)
                    query_nr_results = len(r_json['items'])
                    # print(json.dumps(r_json, indent=4))
                    # add videoIds to storage
                    add_to_storage(storage_, r_json, processed_videoIds, potential_duplicates)
                    print(f'[+] page {page+1}, {year-1} - {year}, query: {query_nr_results}; total: {len(processed_videoIds)}')

                # no next page key found - reached the end
                except KeyError:
                    break
    else:
        storage_ = {'videoIds': PROCESS_SPECIFIC_IDS}
        processed_videoIds = PROCESS_SPECIFIC_IDS
        potential_duplicates = len(processed_videoIds) - len(set(PROCESS_SPECIFIC_IDS))

    # end - write to output json file
    with open(output_filepath, 'wt', encoding='utf-8') as f:
        f.write(json.dumps(storage_, indent=2))
    print(f'[*] done. returned {len(processed_videoIds)} videos\n[*] duplicates: {potential_duplicates}')


def get_videos_info(q, api_key, STORAGE_PATH, NUMBER_OF_VIDEOS='all'):
    '''
    download metadata for all youtube videoIds from get_results()
    :param q:
    :param api_key:
    :return:
    '''
    print(f'--------------------------------------------------------------')
    print(f'[*] Fetching video metadata.')
    # define storage for query and resulting videoId's
    storage_ = {}
    # define output filename based on query
    input_filepath = os.path.join(STORAGE_PATH, f'{q.replace(" ", "_")}_videoIds.json')
    output_filepath = os.path.join(STORAGE_PATH, f'{q.replace(" ", "_")}_videoInfo.json')
    log_filepath = os.path.join(STORAGE_PATH, f'{q.replace(" ", "_")}_log.txt')
    # check if an output file already exists, from e.g. a previous retrieval attempt
    if os.path.isfile(output_filepath):
        # storage will be populated with the existing data
        with open(output_filepath, 'rt', encoding='utf-8') as f_existingdata:
            storage_ = json.load(f_existingdata)
        print(f'[!] output-file already exists. Trying to continue where it left off...')
        # read the last line of the log file and retrieve the videoId from which to continue
        last_logged_videoId = open(log_filepath, 'rt', encoding='utf-8').readlines()[-1].rstrip('\n')
        # check next videoId in videoIds file
        with open(input_filepath, 'rt', encoding='utf-8') as f:
            videoIDs = json.load(f)['videoIds']
            if NUMBER_OF_VIDEOS != 'all':
                print(f'[+] get metadata only for {NUMBER_OF_VIDEOS} of all {len(videoIDs)} videos')
                videoIDs = videoIDs[:NUMBER_OF_VIDEOS]

            last_logged_videoId_idx = videoIDs.index(last_logged_videoId)
            videoId_to_continue_idx = last_logged_videoId_idx + 1
            # check if the next indexv corresponds already to the last item - meaning all data is there
            if len(videoIDs) == videoId_to_continue_idx:
                print(f'[*] all metadata already acquired.')
                return None
            else:
                videoId_to_continue = videoIDs[videoId_to_continue_idx]
        print(f'[*] continue with videoId: {videoId_to_continue}')
        data_left = videoIDs[videoId_to_continue_idx:]
        for index, video_id in enumerate(data_left['videoIds'], 1):
            url = f'https://youtube.googleapis.com/youtube/v3/videos?part=snippet&&id={video_id}&key={api_key}'
            r_json = get_url(url)
            # print(json.dumps(r_json, indent=4))
            storage_[video_id] = r_json['items'][0]['snippet']
            print(f'[*] metadata {index} of {len(data_left["videoIds"])} fetched')
            # write to output file after each videoId retrieved, if search breaks, continue where left off
            with open(output_filepath, 'wt', encoding='utf-8') as f:
                f.write(json.dumps(storage_, indent=2))
            # write processed videoId to log
            with open(log_filepath, 'at', encoding='utf-8') as f:
                f.write(f'{video_id}\n')

    else:
        # iterate over videoIds in output file of get_results()
        with open(input_filepath, 'rt', encoding='utf-8') as f:
            videoIDs = json.load(f)['videoIds']
        if NUMBER_OF_VIDEOS != 'all':
            print(f'[+] get metadata only for {NUMBER_OF_VIDEOS} of all {len(videoIDs)} videos')
            videoIDs = videoIDs[:NUMBER_OF_VIDEOS]

        for index, video_id in enumerate(videoIDs, 1):
            url = f'https://youtube.googleapis.com/youtube/v3/videos?part=snippet&&id={video_id}&key={api_key}'
            r_json = get_url(url)
            # print(json.dumps(r_json, indent=4))
            storage_[video_id] = r_json['items'][0]['snippet']
            print(f'[*] metadata {index} of {len(videoIDs)} fetched')
            # write to output file after each videoId retrieved, if search breaks, continue where left off
            with open(output_filepath, 'wt', encoding='utf-8') as f:
                f.write(json.dumps(storage_, indent=2))
            # write processed videoId to log
            with open(log_filepath, 'at', encoding='utf-8') as f:
                f.write(f'{video_id}\n')

if __name__ == '__main__':
    query = 'Paris walk'
    get_results(query, api_key)
    get_videos_info(query)