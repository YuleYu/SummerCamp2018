import xml.dom.minidom

def ReadXML(path):
    dom = xml.dom.minidom.parse(path)

    root = dom.documentElement


    startTime = root.getElementsByTagName('startTime')[0].childNodes[0].data
    endTime = root.getElementsByTagName('endTime')[0].childNodes[0].data
    imgPath = root.getElementsByTagName('imgPath')[0].childNodes[0].data
    size_x = root.getElementsByTagName('size_x')[0].childNodes[0].data
    size_y = root.getElementsByTagName('size_y')[0].childNodes[0].data
    colSize = root.getElementsByTagName('colSize')[0].childNodes[0].data
    fps = root.getElementsByTagName('fps')[0].childNodes[0].data

<<<<<<< HEAD
    startFrame = int(root.getElementsByTagName('startHOGpic')[0].childNodes[0].data)
    endFrame = int(root.getElementsByTagName('endHOGpic')[0].childNodes[0].data)
    showMidResult = bool(int(root.getElementsByTagName('midResult')[0].childNodes[0].data))
    video_choice = root.getElementsByTagName('videoName')[0].childNodes[0].data

    return startFrame, endFrame, showMidResult, video_choice
=======

    return imgPath, size_x, size_y, startTime, endTime, colSize, fps
>>>>>>> f93167f5d9ca7b16d21a1a41d39b9f8b11f4faf4
