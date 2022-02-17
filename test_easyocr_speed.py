import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import time
import easyocr
import numpy as np



# VIDEO_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\input_videos\5min_test_geneva.mp4"
# IMAGE_PATH = r"C:\Users\mhartman\PycharmProjects\transportation_mode_detection_Yolov5\attention_ocr_fsns\1020p_image_test.jpg"

VIDEO_PATH = "./input_videos/20211229_geneva_old_town_1020p.mp4"
IMAGE_PATH = "1020p_image_test.jpg"


# test on image or video
test = 'image'

ocr_reader = easyocr.Reader(['fr', 'en'], gpu=True)

if test == 'image':
    start = time.time()
    img0 = np.array(cv2.imread(IMAGE_PATH))
    ocr_result = ocr_reader.readtext(img0, detail=1)
    print(f'[*] image took {round(time.time() - start)}s')


if test == 'video':
    cap = cv2.VideoCapture(VIDEO_PATH)
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # dim = (384, 640)
    for frame in range(1, nframes):
        start = time.time()
        ret_val, img0 = cap.read()
        cv2.imwrite("./video_frame.jpg", img0)
        # # resize image
        # img0 = cv2.resize(img0, dim, interpolation=cv2.INTER_AREA)
        ocr_result = ocr_reader.readtext(img0, detail=1)
        print(f'[*] frame {frame} of {nframes} - {round(time.time()-start)}s')