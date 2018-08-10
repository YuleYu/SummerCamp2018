import cv2 as cv
from HOG.ext.get_region import GetRegion
import HOG.ext.ReadXML
import HOG.ext.functions as func
import HOG.ext.hog_compute
import HOG.ext.sampling
import numpy as np

script_path = './script/'
mainScript = script_path + 'main_HOG.xml'
startHOGpic, endHOGpic, showMidResult, video_choice, get_region, pic_path, video_path, result_path = HOG.ext.ReadXML.ReadMainXML(mainScript)
# ========= setting ====================
#video_choice = 'video1'
# script_name = 'main_HOG.xml'
# video_name = 'video1.mp4'
# pic_name = 'video1.png'
# ======================================

pic_name = '{0}.png'.format(video_choice)
script_name = '{0}.xml'.format(video_choice)
video_name = '{0}.mp4'.format(video_choice)

# read in script
script = "{0}{1}".format(script_path, script_name)
pic = "{0}{1}".format(pic_path, pic_name)
video = "{0}{1}".format(video_path, video_name)
imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin = HOG.ext.ReadXML.ReadXML(script)


# get_region
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


packed_img = cv.imread(pic, 0)

# 读入正样本，正样本具体由手工筛选，保存在 同名文件_pos.txt 中
f_positive = open(result_path + video_choice +'_pos.txt','r')
s_pos = []
for entry in f_positive:
    f_beg,f_end = list(map(int,entry.split()))
    for i in range(f_beg,f_end+1):
        s_pos.append((i,func.GetOneFrame(packed_img,size_y,size_x,i,colSize)))
f_positive.close()

hog_pos = func.CalcSample(s_pos,1)

#提取负样本,暂定从f_beg向后取5~50内的帧,可形成405~4050个大小的负样本集合
#用neg_per_shot控制一次假进球帧后面采集多少负样本,neg_per_shot越大,静止帧数越多
f_neg = open(result_path + video_choice +'_neg.txt','r')
s_neg = []
for entry in f_neg:
    f_beg,f_end = list(map(int,entry.split()))
    # print("%d~%d is processing:\n"%(f_beg,f_end))
    for i in range(f_beg,f_beg+5):
        s_neg.append((i,func.GetOneFrame(packed_img,size_y,size_x,i,colSize)))
f_neg.close()
total_neg = len(s_neg)
print(total_neg)

hog_neg = func.CalcSample(s_neg,0)

#计算正样本中心center_pos
center_pos = sum(hog_pos)/hog_pos.shape[0]

#计算正负样本到center的距离
dist_pos = np.zeros((hog_pos.shape[0]))
for i in range(hog_pos.shape[0]):
    dist_pos[i] = np.sqrt(sum((hog_pos[i]-center_pos).reshape(hog_pos[i].size)**2))
dist_neg = np.zeros((hog_neg.shape[0]))
for i in range(hog_neg.shape[0]):
    dist_neg[i] = np.sqrt(sum((hog_neg[i]-center_pos).reshape(hog_neg[i].size)**2))

h = HOG.ext.hog_compute.LoadHOG(result_path + video_choice)  # open .hog file
label = HOG.ext.sampling.GenLabel((result_path + video_choice), (endTime - startTime) * fps)
train_id,test_id = HOG.ext.sampling.HoldOut((endTime - startTime) * fps)
