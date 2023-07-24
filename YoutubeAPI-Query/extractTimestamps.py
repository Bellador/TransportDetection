import os
import re
import json
import copy



def tempony_and_weather_extraction(query, STORAGE_PATH):
    '''
    (1) extract tempony from video description (with regex):
        (1.1) video timestamps and their content [new key: extractedVideoTimestamps]
        (1.2) dates (when the video was actually recorded) [new key: extractedDates]
        (1.3) times (when the video was actually recorded) [new key: extractedTime]

    (2) extract weather conditions from video description and TITLE (with regex) weather condition based on textual descriptions
        (2.1) good weather word lists: sunny, heat, hot
        (2.2) bad weather word lists: rain*, windy, murky, humid, hail, bleak, gusty, misty

    and then add the extracted temporal information back into the json
    :return:
    '''
    # adapt input/output filenames based on query
    input_filepath = os.path.join(STORAGE_PATH, f'{query}_videoInfo.json')
    output_filepath = os.path.join(STORAGE_PATH, f'{query}_videoInfoExtracted.json')
    with open(input_filepath, 'rt', encoding='utf-8') as f:
        data_dict = json.load(f)
    # REGEX TEMPONY -------------------------------------------------------------------
    # extract the timestamp and the text behind it (describing the video timestamp)
    # matches: (1) timestamp, (2) text that follows after it (e.g. '10:01:30 RUE DE LA MICHODIERE \n')
    video_timestamp_pattern = r'(\d{0,2}[:]*\d{1,2}[:]+\d{2,4})(?!.* AM)(?!.* PM)\s+(.+)\n' # does not match AM, PM instances that point at mentions of time
    # pattern to extract the timestamp e.g. when the video was filmed (according to the author)
    # matches: normal dates (e.g. FILMED ON 09/04/2022 or FILMED ON 04.22)
    date_timestamp_pattern = r'\d{2}[\.\/]*\d{2}[\.\/]+\d{2,4}'
    # extracts day time e.g. when the video was filmed (according to the author)
    # matches: mentions of day time (e.g. 10:30 AM)
    time_timestamp_pattern = r'(\d{1,2}[:]*\d{0,2})\s+(AM|PM)'
    # REGEX WEATHER  -------------------------------------------------------------------
    good_weather_pattern = r'([ ]*good weather[ ]*|[ ]*nice weather[ ]*|[ ]*clear sky[ ]*|[ ]*blue sky[ ]*|[ ]*hot weather[ ]*|[ ]*hot day[ ]*| sunny | sun | heat | cloudless | windless )'
    bad_weather_pattern = r'([ ]*dark sky[ ]*|[ ]*bad weather[ ]*| rain[y]{0,1} | wind[y]{0,1} | gust[y]{0,1} | snow[y]{0,1} | hail )'

    # make a deep copy of the original data, to be added with extracted matches
    data_dict_copy = copy.deepcopy(data_dict)
    # store videoIds with detected manually added timestamps
    videoIDs_with_timestamps = []
    # iterate over the video descriptions
    for index, (videoID, value) in enumerate(data_dict.items(), 1):
        # check to see if the key is a videoId
        if videoID != 'query':
            description = value['description']
            title = value['title']
            # PREPROCESSING
            # remove urls
            description = re.sub(r'http\S+', '', description)

            # 1. match video timestamps
            matches = re.findall(video_timestamp_pattern, description)
            # 1.1 add matches (even if empty) to data copy
            data_dict_copy[videoID]['extractedVideoTimestamps'] = matches
            if matches:
                # add videoID to track list if matches were found
                videoIDs_with_timestamps.append(videoID)

            # 1.2 match dates
            matches = re.findall(date_timestamp_pattern, description)
            # add matches (even if empty) to data copy
            data_dict_copy[videoID]['extractedDates'] = matches

            # 1.3 match day times
            matches = re.findall(time_timestamp_pattern, description)
            # add matches (even if empty) to data copy
            data_dict_copy[videoID]['extractedTimes'] = matches

            # 2.1 match good weather
            matches_title = re.findall(good_weather_pattern, title, re.IGNORECASE)
            matches_description = re.findall(good_weather_pattern, description, re.IGNORECASE)
            matches = matches_title + matches_description
            # add matches
            data_dict_copy[videoID]['goodWeather'] = matches

            # 2.2 match bad weather
            matches_title = re.findall(bad_weather_pattern, title, re.IGNORECASE)
            matches_description = re.findall(bad_weather_pattern, description, re.IGNORECASE)
            matches = matches_title + matches_description
            # add matches
            data_dict_copy[videoID]['badWeather'] = matches


    with open(output_filepath, 'wt', encoding='utf-8') as f:
        f.write(json.dumps(data_dict_copy, indent=2))

    return videoIDs_with_timestamps


if __name__ == '__main__':

    query = 'Paris_walk'
    STORAGE_PATH = r"C:\Users\mhartman\PycharmProjects\TransportDetectionV3\manual_video_analysis"
    tempony_and_weather_extraction(query, STORAGE_PATH)
