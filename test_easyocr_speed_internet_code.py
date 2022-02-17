# https://youtu.be/owiqdzha_DE
"""
pip install easyocr
https://github.com/JaidedAI/EasyOCR
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import cv2
import time
import easyocr



# For the first time it downloads the models for the languages chosen below.
# Not all languages are compatible with each other so you cannot put
# multiple languages below
# reader = easyocr.Reader(['hi', 'te', 'en'])  #Hindi, telugu, and English
# The above gives error that Telugu is only compatible with English.

# So let us just use Hindi and English
# To use GPU you need to have CUDA configured for the pytorch library.
reader = easyocr.Reader(['fr', 'en'], gpu=False)  # Hindi, telugu, and English
start = time.time()
img = cv2.imread('video_frame.jpg')

results = reader.readtext(img, detail=1, paragraph=False)  # Set detail to 0 for simple text output
print(f'[*] time: {round(time.time() - start)}s')
# Paragraph=True will combine all results making it easy to capture it in a dataframe.


# To display the text on the original image or show bounding boxes
# we need the coordinates for the text. So make sure the detail=1 above, readtext.
# display the OCR'd text and associated probability
for (bbox, text, prob) in results:
    # Define bounding boxes
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))

    # Remove non-ASCII characters to display clean text on the image (using opencv)
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()

    # Put rectangles and text on the image
    cv2.rectangle(img, tl, br, (0, 255, 0), 2)
    cv2.putText(img, text, (tl[0], tl[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
# show the output image
cv2.imshow("Image", img)
cv2.waitKey(0)