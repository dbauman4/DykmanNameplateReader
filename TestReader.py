import cv2
import matplotlib.pyplot as plt
import pytesseract
import numpy as np
from scipy import ndimage
from scipy.ndimage import interpolation as inter

def findScore(img, angle):
    data = inter.rotate(img, angle, reshape = False, order = 0)
    hist = np.sum(data, axis = 1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

def skewCorrect(img):
    img = cv2.resize(img, (0, 0), fx = 1, fy = 1)

    delta = 1
    limit = 45
    angles = np.arange(-limit, limit+delta, delta)
    scores = []
    for angle in angles:
        hist, score = findScore(img, angle)
        scores.append(score)
    bestScore = max(scores)
    bestAngle = angles[scores.index(bestScore)]
    rotated = inter.rotate(img, bestAngle, reshape = False, order = 0)
    return rotated

image = cv2.imread("Motor-Nameplate-3.jpg")
plt.imshow(image)
plt.show()

rotated = skewCorrect(image)
plt.imshow(rotated)
plt.show()
extract = pytesseract.image_to_string(rotated)
print(extract)

grayImage = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
plt.imshow(grayImage)
plt.show()
extract = pytesseract.image_to_string(grayImage)
print(extract)

(thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 91, 255, cv2.THRESH_TOZERO)
plt.imshow(blackAndWhiteImage)
plt.show()
extract = pytesseract.image_to_string(blackAndWhiteImage)
print(extract)
