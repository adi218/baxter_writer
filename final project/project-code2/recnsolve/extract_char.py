import numpy as np
import cv2
import matplotlib.pyplot as plt


def segment(img, mask_color):
    pix = len(img[0, :, 0])
    image_num = 0
    while pix > 40:
        for i in range(len(img[0, :, 0]) - 40):
            i = i + 40
            k = 0
            for j in range(len(img[:, i, 0])):
                less = np.less(img[j, i, :], mask_color)
                if np.all(less):
                    k = i
            if k == 0:
                image_num += 1
                img1 = img[:, 0:i + 10]
                img1 = cv2.resize(img1, (45, 45), interpolation=cv2.INTER_AREA)
                filename = str(image_num) + ".png"
                cv2.imwrite(filename, img1)
                img = img[:, i + 10:]
                break
        pix = len(img[0, :, 0])

    return image_num


def image_segmentation(img):
    # print(img)
    img = img[200:500, :]
    plt.imshow(img)
    plt.show()
    mask_color = [50, 50, 50]
    k = 0
    for i in range(len(img[:, 0, 0])):
        for j in range(len(img[i, :, 0])):
            less = np.less(img[i, j, :], mask_color)
            if np.all(less):
                # print("enter")
                k = 1
                img = img[i-12:, :]
                break
        if k == 1:
            break
    # plt.imshow(img)
    # plt.show()
    k = 0
    for i in range(len(img[:, 0, 0])):
        i = (len(img[:, 0, 0]) - 1) - i
        for j in range(len(img[i, :, 0])):
            less = np.less(img[i, j, :], mask_color)
            if np.all(less):
                # print("enter")
                k = 1
                img = img[0:i+12, :]
                break
        if k == 1:
            break
    # plt.imshow(img)
    # plt.show()
    k = 0
    for i in range(len(img[0, :, 0])):
        for j in range(len(img[:, i, 0])):
            less = np.less(img[j, i, :], mask_color)
            if np.all(less):
                # print("enter")
                # print(i)
                k = 1
                img = img[:, i-12:]
                break
        if k == 1:
            break

    # plt.imshow(img)
    # plt.show()
    k = 0
    for i in range(len(img[0, :, 0])):
        i = (len(img[0, :, 0]) - 1) - i
        for j in range(len(img[:, i, 0])):
            less = np.less(img[j, i, :], mask_color)
            if np.all(less):
                k = 1
                print("enter")
                img = img[:, 0:i+12]
                break
        if k == 1:
            break

    total_images = segment(img, mask_color)

    return total_images
