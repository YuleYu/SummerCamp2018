import xml.dom.minidom

def readXML(path):
    dom = xml.dom.minidom.parse(path)

    root = dom.documentElement


    startTime = root.getElementsByTagName('startTime')[0].childNodes[0].data
    endTime = root.getElementsByTagName('endTime')[0].childNodes[0].data
    imgPath = root.getElementsByTagName('imgPath')[0].childNodes[0].data
    size_x = root.getElementsByTagName('size_x')[0].childNodes[0].data
    size_y = root.getElementsByTagName('size_y')[0].childNodes[0].data
    colSize = root.getElementsByTagName('colSize')[0].childNodes[0].data
    fps = root.getElementsByTagName('fps')[0].childNodes[0].data


    return imgPath, size_x, size_y, startTime, endTime, colSize, fps