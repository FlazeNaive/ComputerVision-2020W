import cv2 as cv
import os
import glob

from PIL import Image

PATH = 'BioFaceDatabase\BioID-FaceDatabase-V1.2\wait'

files = os.listdir(PATH)
cnt = len(files)
paths = glob.glob(os.path.join(PATH, '*.jpg'))

i = 0
for path in paths:
    img = Image.open(path)
    i = i + 1
    img.show()
    img.save(str(i) +'.pgm')
