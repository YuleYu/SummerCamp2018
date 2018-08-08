import numpy as np
import os
import struct

# def LoadMnist_all():
#     s = []
#     path = './mnist'
#     files = os.listdir(path)
#     for file in files:
#         if not os.path.isdir(file):
#             f = open(path+'/'+file)


def LoadMnistImage(path):
    f = open(path, 'rb').read()
    head = struct.unpack_from('>IIII', f, 0)

    offset = struct.calcsize('>IIII')
    imgNum = head[1]
    width = head[2]
    height = head[3]
    # [60000]*28*28
    bits = imgNum * width * height
    bitsString = '>' + str(bits) + 'B'
    image = struct.unpack_from(bitsString, f, offset)
    image = np.reshape(image, [imgNum, width, height])
    return image


def LoadMnistLabel(path):
    f = open(path, 'rb').read()
    head = struct.unpack_from('>II', f, 0)

    offset = struct.calcsize('>II')
    imgNum = head[1]

    offset = struct.calcsize('>II')
    numString = '>'+ str(imgNum) + 'B'
    label = struct.unpack_from(numString, f, offset)
    label = np.reshape(label, [imgNum, 1])
    return label