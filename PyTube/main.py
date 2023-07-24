import os
from pytube import YouTube

# -------------- CHANGE HERE -------------------
PROJECT_NAME = 'Geneva_virtual_walk_videos' # generates a project folder that stores all videos from the same session
MAX_RESOLUTION = '1080p' # will select the stream that comes closest to this parameter but will not exceed it
MIME_TYPE = 'video/mp4'
VIDEO_AND_AUDIO = False  # sole video streams are available in higher resolution. Coupled video and audio streams are capped at 720p!
GENEVA_LINKS = [
    'https://www.youtube.com/watch?v=rvAUqPkbhq0', # good
    'https://www.youtube.com/watch?v=sfklze6kYw4', # good
    'https://www.youtube.com/watch?v=Bc3TnAk-2Pw', # good
    'https://www.youtube.com/watch?v=4CnzKD3PYyw', # good
    'https://www.youtube.com/watch?v=vl2sfyFIDU0', # very good, even labelled video sections
    'https://www.youtube.com/watch?v=wKreSUENeIg', # good but not linear, but labelled video sections
    'https://www.youtube.com/watch?v=APoJVvxAAWI', # ok ish, lake walk
    'https://www.youtube.com/watch?v=u0M9ue0J0rk', # good
    'https://www.youtube.com/watch?v=SslvMScNvvE', # good
    'https://www.youtube.com/watch?v=iJaXlw3eHWI', # good
    'https://www.youtube.com/watch?v=Nr6eNR3ddRs' # good
]

LINKS = GENEVA_LINKS
# -----------------------------------------------

# create project folder (if not present) in local directory for video storing
VIDEO_SAVE_PATH = f'./{PROJECT_NAME}'
if not os.path.isdir(VIDEO_SAVE_PATH):
    os.mkdir(VIDEO_SAVE_PATH)
    print(f'[+] created project folder: {VIDEO_SAVE_PATH}')
else:
    print(f'[*] project folder: {VIDEO_SAVE_PATH} already exists.')

# check for duplicate links
link_list_len = len(LINKS)
LINKS = set(LINKS)
link_set_len = len(LINKS)
print(f'[*] {link_list_len - link_set_len} duplicates were found - handled.')

for index, link in enumerate(LINKS, 1):
    print(f'[+] -------- video {index} of {link_set_len} -------------')
    yt = YouTube(link)
    avail_res_and_mime = [(stream.resolution, stream.mime_type) for stream in yt.streams.filter(progressive=VIDEO_AND_AUDIO)] #  progressive = True (searches for audio and video stream)

    res_numbs_with_correct_mime = [int(res_mime[0][:-1]) for res_mime in avail_res_and_mime if res_mime[1] == MIME_TYPE]
    highest_pos_res = 0
    for res in res_numbs_with_correct_mime:
        if res <= int(MAX_RESOLUTION[:-1]) and res > highest_pos_res:
            highest_pos_res = res
    print(f'        [*] available res and MIME:')
    for index, res in enumerate(avail_res_and_mime, 1):
        print(f'                  ({index}) {res[0]}, {res[1]}')
    print(f'        [*] {highest_pos_res}p and {MIME_TYPE} chosen since largest below MAX RES of {MAX_RESOLUTION}')
    print(f'        [+] {link} downloading..')
    target_video = yt.streams.filter(progressive=False, mime_type=MIME_TYPE, res=f'{highest_pos_res}p') \
        .desc() \
        .first()
    output_fileame = "".join(x for x in target_video.default_filename
                             if ord(x) < 128 and
                                x not in '{}[]()`~:;,*"+-!?=$£@¦|').replace(' ', '_')
    target_video.download(VIDEO_SAVE_PATH, filename=output_fileame)
    print(f'        [+] Done. {output_fileame} saved.')
