import xml.dom.minidom


def ReadXML(path: object) -> object:
    dom = xml.dom.minidom.parse(path)

    root = dom.documentElement

    startTime = int(root.getElementsByTagName('startTime')[0].childNodes[0].data)
    endTime = int(root.getElementsByTagName('endTime')[0].childNodes[0].data)
    imgPath = root.getElementsByTagName('imgPath')[0].childNodes[0].data
    size_x = int(root.getElementsByTagName('size_x')[0].childNodes[0].data)
    size_y = int(root.getElementsByTagName('size_y')[0].childNodes[0].data)
    colSize = int(root.getElementsByTagName('colSize')[0].childNodes[0].data)
    fps = int(root.getElementsByTagName('fps')[0].childNodes[0].data)
    bin = int(root.getElementsByTagName('bin')[0].childNodes[0].data)

    return imgPath, size_x, size_y, startTime, endTime, colSize, fps, bin

def ReadMainXML(path: object) -> object:
    dom = xml.dom.minidom.parse(path)
    root = dom.documentElement

    startFrame = int(root.getElementsByTagName('startHOGpic')[0].childNodes[0].data)
    endFrame = int(root.getElementsByTagName('endHOGpic')[0].childNodes[0].data)
    showMidResult = bool(int(root.getElementsByTagName('midResult')[0].childNodes[0].data))
    video_choice = root.getElementsByTagName('videoName')[0].childNodes[0].data

    return startFrame, endFrame, showMidResult, video_choice