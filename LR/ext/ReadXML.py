import xml.dom.minidom


def ReadMainXML(path: object) -> object:
    dom = xml.dom.minidom.parse(path)

    root = dom.documentElement

    trainingSet = int(root.getElementsByTagName('trainingSet')[0].childNodes[0].data)
    learningRate = int(root.getElementsByTagName('learningRate')[0].childNodes[0].data)
    batchSize = root.getElementsByTagName('batchSize')[0].childNodes[0].data
    continuousFrame = int(root.getElementsByTagName('continuousFrame')[0].childNodes[0].data)
    lineSearch = int(root.getElementsByTagName('lineSearch')[0].childNodes[0].data)
    weightDecay = int(root.getElementsByTagName('colSize')[0].childNodes[0].data)

    return trainingSet, learningRate, batchSize, continuousFrame, lineSearch, weightDecay