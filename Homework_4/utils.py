import numpy as np
import cv2
import matplotlib.pyplot as plt


def imshow(img):
    """
    Shortaround for showing color images with matplotlib
    :param img: 
    :return: 
    """
    converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(converted, cmap='gray')
    plt.show()


def addPoints(img, pts):
    """
    Add points to image as overlays.
    :param img: reference image.
    :param pts: points to be applied to image.
    :return: image with points overlays.
    """
    return cv2.polylines(np.copy(img), [pts], isClosed=True, color=(0, 0, 255))


def warpImage(img, srcPts, dstPts, dsize):
    """
    Apply homography that maps srcPts to dstPts to specified image.
    :param img: image to be warped.
    :param srcPts: landmark points in source image.
    :param dstPts: landmark points in destination image.
    :param dsize: viewport size relative to destination image viewport.
    :return: warped image.
    """
    m, status = cv2.findHomography(srcPts.astype('float32'), dstPts.astype('float32'))
    return cv2.warpPerspective(img, m, dsize)


def translate(img, offset, dsize=None):
    """
    Translate image by specified amount.
    :param img: image to be translated.
    :param offset: translation amount.
    :param dsize: viewport size relative to initial viewport.
    :return: translated image.
    """

    if dsize is None:
        dsize = (img.shape[1], img.shape[0])

    m = np.float32([[1, 0, offset[0]], [0, 1, offset[1]]])
    return cv2.warpAffine(img, m, dsize)
