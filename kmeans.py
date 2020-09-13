import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from sklearn.datasets.samples_generator import make_blobs
import random
import time

img = imread("porsche911r.png")
print(img.shape)
plt.imshow(img)
plt.show()

rows = img.shape[0]
cols = img.shape[1]
t0 = time.time()

k = 10
clusterCenters = []
clusters = []
points, categories = make_blobs(n_samples = 100, centers = k, n_features = 2, random_state = 0, cluster_std = 0.5)

def distance3D(p1, p2):
    return np.sqrt(np.square(p1[0]-p2[0])+np.square(p1[1]-p2[1])+np.square(p1[2]-p2[2]))


def closestCenters(clusterCenters):
    clusters = []
    for n in range(k):
        clusters.append([])
    i = 0
    newImg = []
    #Cant use img
    for row in img:
        newImgRow = []
        for pixel in row:
            minDistance = 0
            closestCenter = 10000000
            index = 0
            for x in range(len(clusterCenters)):
                #cant use just pixel, use coordinates of pixel.
                distance = distance3D(pixel, clusterCenters[x])
                if distance < closestCenter:
                    closestCenter = distance
                    index = x
            newImgRow.append(index)
            clusters[index].append(pixel)
        newImg.append(newImgRow)

    # print(len(clusters))
    newClusterCenters = getNewClusters(clusters)
    contouredImage = createImage(newImg, newClusterCenters)

    return contouredImage, newClusterCenters


def createImage(newImgIndexes, newClusterCenters):
    createdImage = []
    for row in newImgIndexes:
        renderRow = []
        for index in row:
            renderRow.append(newClusterCenters[index])
        createdImage.append(renderRow)

    return createdImage



def getNewClusters(clusters):
    for cluster in clusters:
        center = []
        averageR = []
        averageG = []
        averageB = []
        for p in cluster:
            averageR.append(p[0])
            averageG.append(p[1])
            averageB.append(p[2])
        center = [np.average(averageR), np.average(averageG), np.average(averageB)]
        newClusterCenters.append(center)
    return newClusterCenters
# def renderImage(l1, l2):

def initClusters():
    for n in range(k):
        randomRgb = img[random.randint(0, rows + 1)][random.randint(0, cols + 1)]
        clusterCenters.append(randomRgb)
    return clusterCenters

# p = percent margin
def withinMargin(l1, l2, p):
    if (l1 == [] or l2 == []):
        return False
    lowerBound = (100.0 - p)/100
    upperBound = (100 + p)/100
    for x in range(k):
        for y in range(3):
            c1 = l1[x][y]
            c2 = l2[x][y]
            low = c1 * lowerBound
            high = c1 * upperBound
            if not (low <= c2 and high >= c2):
                return False
    return True



clusterCenters = initClusters()
newClusterCenters = []
iteration = 0

while not withinMargin(clusterCenters, newClusterCenters, 5):
    print("EPOCH ", iteration)
    if (newClusterCenters != []):
        clusterCenters = newClusterCenters
        newClusterCenters = []
    print(clusterCenters)
    imageToRender, newClusterCenters = closestCenters(clusterCenters)
    iteration += 1



    # print(len(clusters))
# print(imageToRender)
print("K-Value:")
print(k)
print("Image Dimentions:")
print(rows, cols)
print("Final Cluster Centers:")
print(newClusterCenters)
print("Number of Epochs Run:")
print(iteration)
t1 = time.time()
print("Run Time:")
print(t1 - t0)
plt.imshow(imageToRender)
plt.show()

# print(clusterCenters)
