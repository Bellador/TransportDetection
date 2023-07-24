import os
from pathlib import Path
from pytube import YouTube

def ytvideo_download(LINKS, PROJECT_NAME, ROOT_PATH,
                     MAX_RESOLUTION='720p',
                     MIME_TYPE='video/mp4',
                     ONLY_VIDEO=False):

    # create project folder (if not present) in local directory for video storing
    VIDEOS_SAVE_PATH = os.path.join(ROOT_PATH, 'input_videos', PROJECT_NAME)
    Path(VIDEOS_SAVE_PATH).mkdir(parents=True, exist_ok=True)
    # check for duplicate links
    link_list_len = len(LINKS)
    LINKS = set(LINKS)
    link_set_len = len(LINKS)
    print(f'[*] {link_list_len - link_set_len} duplicates were found - handled.')

    for index, link in enumerate(LINKS, 1):
        print(f'[+] -------- video {index} of {link_set_len} -------------')
        # output filename will be the unique videoID
        output_filename = link.split('=')[-1] + '.mp4'
        # check if video already downloaded
        if os.path.isfile(os.path.join(VIDEOS_SAVE_PATH, output_filename)):
            print(f'[*] {output_filename} already downloaded.')
            continue
        yt = YouTube(link)
        avail_res_and_mime = [(stream.resolution, stream.mime_type) for stream in yt.streams.filter(progressive=True, mime_type=MIME_TYPE, only_video=ONLY_VIDEO)] #  progressive = True (searches for audio and video stream)
        res_numbs_with_correct_mime = [int(res_mime[0][:-1]) for res_mime in avail_res_and_mime if res_mime[1] == MIME_TYPE]
        highest_pos_res = 0
        for res in res_numbs_with_correct_mime:
            if res <= int(MAX_RESOLUTION[:-1]) and res > highest_pos_res:
                highest_pos_res = res
        # code to print MIME types of video available for download
        print(f'        [*] available RES and MIME:')
        for res_index, res in enumerate(avail_res_and_mime, 1):
            print(f'                  ({res_index}) {res[0]}, {res[1]}')
        print(f'        [*] {highest_pos_res}p and {MIME_TYPE} chosen since largest below MAX RES of {MAX_RESOLUTION}')
        print(f'        [+] {link} downloading..')
        target_video = yt.streams.filter(progressive=True, mime_type=MIME_TYPE, res=f'{highest_pos_res}p') \
            .desc() \
            .first()
        if target_video is not None:
            max_tries = 3
            tries = 0
            while (tries < max_tries):
                try:
                    target_video.download(VIDEOS_SAVE_PATH, filename=output_filename)
                    print(f'        [+] Done. {output_filename} saved.')
                    break
                except Exception as e:
                    tries += 1
                    print(f'[!!] video download error: {e}')
                    print(f'[!!] try {tries} of {max_tries}')

            # delete unfinished file if download failed
            if tries == max_tries:
                FILE_PATH = os.path.join(VIDEOS_SAVE_PATH, output_filename)
                if os.path.isfile(FILE_PATH):
                    os.remove(os.path.join(VIDEOS_SAVE_PATH, output_filename))
                    print(f'[!!] failed download file {output_filename} removed')
        else:
            print(f'[!] no video matching the requirements:\n  link:{link}\n  max resolution:{MAX_RESOLUTION}\n  only video no audio:{ONLY_VIDEO}')


    return VIDEOS_SAVE_PATH


if __name__ == '__main__':
    # -------------- CHANGE HERE -------------------

    PROJECT_NAME = 'Paris_walk_predownload'
    ROOT_PATH = '../input_videos'
    MAX_RESOLUTION = '720p'  # will select the stream that comes closest to this parameter but will not exceed it
    MIME_TYPE = 'video/mp4'
    ONLY_VIDEO = False  # sole video streams are available in higher resolution. Coupled video and audio streams are capped at 720p!

    with open('../specific_videoids_to_process.txt') as file:
        SPECIFIC_VIDEOIDS_TO_PROCESS = [line.rstrip() for line in file]

    LINKS = [f'https://www.youtube.com/watch?v={videoID}' for videoID in SPECIFIC_VIDEOIDS_TO_PROCESS]
    # -----------------------------------------------
    ytvideo_download(LINKS, PROJECT_NAME, ROOT_PATH,
                MAX_RESOLUTION=MAX_RESOLUTION,
                MIME_TYPE=MIME_TYPE,
                ONLY_VIDEO=ONLY_VIDEO)
