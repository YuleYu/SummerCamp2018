import HOG.ext.hog_compute as hog_compute
import HOG.ext.sampling as sampling
import HOG.ext.ReadXML
import pickle

home = 'C:/Users/peter/Documents/GitHub/SummerCamp2018/'
videochoice = "video1"
fname = home + 'HOG/result/' + videochoice
imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin = HOG.ext.ReadXML.ReadXML(
    home + 'HOG/script/' + videochoice + '.xml')
n_frames = (endTime - startTime) * fps

h = hog_compute.LoadHOG(fname)
label = sampling.GenLabel(fname, n_frames)

train_id, test_id = sampling.HoldOut(h.shape[0])
x_train = h[train_id]
y_train = label[train_id]
x_test = h[test_id]
y_test = label[test_id]

f_train = open(home+'HOG/result/train.dat','wb')
f_test = open(home+'HOG/result/test.dat','wb')

pickle.dump(x_train,f_train,1)
pickle.dump(y_train,f_train,1)
f_train.close()

pickle.dump(x_test,f_test,1)
pickle.dump(y_test,f_test,1)
f_test.close()
