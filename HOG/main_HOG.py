import cv2 as cv
import matplotlib.pyplot as plt
from math import floor
from numpy import mean
from numpy.matlib import zeros

from HOG.ext.functions import HOGCalc, HOG_pic
from HOG.ext.get_region import GetRegion
import HOG.ext.ReadXML


script_path = './script/'
mainScript = f"{script_path}{'main_HOG.xml'}"
startHOGpic, endHOGpic, showMidResult, video_choice = HOG.ext.ReadXML.ReadMainXML(mainScript)
# ========= setting ====================
#video_choice = 'video1'
# script_name = 'script_example.xml'
# video_name = 'video1.mp4'
# ======================================

pic_path = './pic/'
video_path = './video/'
pic_name = '{0}.png'.format(video_choice)
script_name = '{0}.xml'.format(video_choice)
video_name = '{0}.mp4'.format(video_choice)

# read in script
script = "{0}{1}".format(script_path, script_name)
pic = "{0}{1}".format(pic_path, pic_name)
video = "{0}{1}".format(video_path, video_name)
imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin = HOG.ext.ReadXML.ReadXML(script)


# get_region
get_region = False
if get_region == True:
    drawing = False  # 如果按下鼠标，则为true
    mode = True  # 如果是 True 则画矩形。按 m 键变成绘制曲线。

    cap = cv.VideoCapture(video_path + video_name)
    success, frame = cap.read()
    cv.namedWindow('img')

    GetRegion(frame)
    form = '%5d\t%5d\n'
    f = open(script_path + "target_index.txt", 'w')
    # f.write(form %( ix, x_final))
    # f.write(form % (iy, y_final))
    f.close()

img = cv.imread(pic, 0)
# img_show = zeros((size_y, size_x))
# for i in range(startHOGpic, endHOGpic, 1):
#     raw = floor(i / colSize)
#     col = floor(i % colSize)
#     img_show += img[raw*size_y:raw*size_y+size_y-1, col*size_x:col*size_x+size_x-1]
img_show = img[int(size_y) * startHOGpic:(int(size_y) * endHOGpic - 1), 0:(int(size_x) * int(colSize) - 1)]
hog = HOGCalc(img_show, int(bin))
hog_image = HOG_pic(img_show, hog)

num_col, num_raw = img.shape
num_col /= size_y
num_raw /= size_x

hogDic = []
for count in range(int(num_raw) * int(num_col)):
    raw = floor(count / colSize)
    col = floor(count % colSize)
    assert isinstance(size_y, object)
    hogDic.append(hog[raw * int(size_y):raw * int(size_y) - 1, col * size_x:col * size_x + size_x - 1])

# HOG picture
if showMidResult == True:
    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()

# compute the distance of HOG
mean_value = []
for every_img in hogDic:
    mean_value.append(mean(every_img))



print('over')