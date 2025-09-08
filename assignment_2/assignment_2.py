import cv2 as cv
import numpy as np
import os

def padding(image, border_width):
    img = cv.imread(image)

    if img is None:
        print("Could not open image")
        exit(1)

    reflect = cv.copyMakeBorder(img,border_width, border_width, border_width, border_width, cv.BORDER_REFLECT)
    cv.imshow('reflect', reflect)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('images/reflect_image.png', reflect)
    return reflect

def crop(image, x_0, x_1, y_0, y_1):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    # x_1 = 512 - 130           y_1 = 512 - 130
    cropped_image = img[y_0+80:y_1-130, x_0+80:x_1-130]
    cv.imshow('cropped_image', cropped_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('images/cropped_image.png', cropped_image)
    return cropped_image

def resize(image, width, height):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    resizing = cv.resize(img, (width, height))
    cv.imshow('resize', resizing)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite('images/resized_image.png', resizing)
    return resizing

def copy(image, emptyPictureArray):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    height, width, channel = img.shape

    if emptyPictureArray is None:
        emptyPictureArray = np.zeros((height, width, channel), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            emptyPictureArray[i][j] = img[i][j]

    cv.imshow('copy', emptyPictureArray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("images/copied_image.png", emptyPictureArray)
    return emptyPictureArray


def grayscale(image):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('gray', gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("images/gray_image.png", gray)
    return gray

def hsv(image):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    cv.imshow('hsv', hsv_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("images/hsv_image.png",hsv_img)
    return hsv_img

def hue_shifted(image, emptyPictureArray, hue):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    height, width, channels = img.shape
    if emptyPictureArray is None:
        emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)

    #When we add a value to the RGB components, the result may fall outside the valid range (0–255).
    # Therefore, I use clipping, which means that any values above 255 are set to 255 (maximum intensity),
    # and any values below 0 are set to 0 (minimum intensity). For example: 220 + 50 = 270, but since 270 > 255, the result is clipped to 255.
    shifted = np.clip(img.astype(np.int16) + int(hue), 0, 255)
    shifted = shifted.astype(np.uint8)

    emptyPictureArray[:] = shifted

    cv.imshow("hue_shifted", emptyPictureArray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("images/rgb_shifting_{}_image.png".format(hue), emptyPictureArray)
    return emptyPictureArray


def smoothing(image):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    dst = cv.GaussianBlur(img, (15, 15), cv.BORDER_DEFAULT)
    cv.imshow('gaussian', dst)
    cv.waitKey(0)
    cv.destroyAllWindows()

    cv.imwrite("images/gaussian_smoothing_image.png", dst)
    return dst

def rotation(image, rotation_angle):
    img = cv.imread(image)
    if img is None:
        print("Could not open image")
        exit(1)

    if rotation_angle == 90:
        rotated = cv.rotate(img, cv.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv.rotate(img, cv.ROTATE_180)
    else:
        print("Only 90 or 180 degrees allowed")
        return None

    cv.imshow('rotated', rotated)
    cv.waitKey(0)
    cv.destroyAllWindows()

    out_filename = f"rotated_{rotation_angle}_image.png"
    save_path = os.path.join("images", out_filename)
    cv.imwrite(save_path, rotated)
    return rotated


if __name__ == '__main__':
    padding('lena-2.png', 100)
    crop('lena-2.png', 0, 512, 0, 512)
    resize('lena-2.png', 200, 200)
    copy("lena-2.png", None)
    grayscale('lena-2.png')
    hsv('lena-2.png')
    hue_shifted('lena-2.png', None, 50)
    smoothing('lena-2.png')
    angle = int(input("Enter rotation angle (90 or 180): "))
    rotation("lena-2.png", angle)
