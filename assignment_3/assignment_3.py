import cv2 as cv
import numpy as np


def sobel_edge_function(image):
    img = cv.imread(image)
    if img is None:
        print("Could not load image.")
        exit(1)

    gaussian = cv.GaussianBlur(img, (3, 3), 0)
    sobel = cv.Sobel(src=gaussian, ddepth=cv.CV_64F, dx=1, dy=1, ksize=1)

    cv.imshow('sobel', sobel)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("saved_images/sobel.png", sobel)

def canny_edge_detection(image, threshold_1, threshold_2):
    img = cv.imread(image)
    if img is None:
        print("Could not load image.")
        exit(1)

    gaussian = cv.GaussianBlur(img, (3, 3), 0)
    edges = cv.Canny(gaussian, threshold_1, threshold_2)

    cv.imshow('canny_edge_detection', edges)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("saved_images/canny_edge_detection.png", edges)

def template_match(image, template):
    img_normal = cv.imread(image)
    img_gray = cv.imread(image,0)
    temp_gray = cv.imread(template,0)
    if img_normal is None or img_gray is None or temp_gray is None:
        print("Could not load image(s).")
        exit(1)

    h, w = temp_gray.shape[:2]
    res = cv.matchTemplate(img_gray, temp_gray, cv.TM_CCOEFF_NORMED)
    treshold = 0.9
    loc = np.where(res >= treshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img_normal, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv.imshow('template_match', img_normal)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("saved_images/template_match.png", img_normal)

def resize(image, scale_factor:int, up_or_down:str):
    img = cv.imread(image)
    if img is None:
        print("Could not load image.")
        exit(1)

    height, width = img.shape[:2]

    if up_or_down == 'up':
        img = cv.pyrUp(img, dstsize=(scale_factor * width, scale_factor * height))
        print(f"zoomed in x{scale_factor}.")
        cv.imwrite("saved_images/zoomed_in.png", img)
        cv.imshow('zoomed in', img)
    elif up_or_down == 'down':
        img = cv.pyrDown(img, dstsize=(width // scale_factor, height // scale_factor))
        print(f"zoomed out x{scale_factor}.")
        cv.imwrite("saved_images/zoomed_out.png", img)
        cv.imshow('zoomed down', img)
    else:
        print("Invalid input.")

    cv.imshow('resized', img)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    sobel_edge_function('lambo.png')
    canny_edge_detection('lambo.png', 100, 100)
    template_match('shapes-1.png', 'shapes_template.jpg')

    zoom = str(input("Enter 'up' for zooming in and 'down' for zooming out: "))
    resize('lambo.png',2, zoom)
