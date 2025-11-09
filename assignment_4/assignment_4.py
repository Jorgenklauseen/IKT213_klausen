import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

def harris_corner(reference_image):
    img = cv.imread(reference_image)
    if img is None:
        print("Could not open image")
        exit(1)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    dst = cv.dilate(dst, None)

    img[dst > 0.01*dst.max()] = [0, 0, 255]
    cv.imshow('Harris Corner', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    cv.imwrite("saved_images/harris.png", img)


def sift(image_to_align, reference_image, max_features, good_match_precent):
    img1 = cv.imread(image_to_align, cv.IMREAD_GRAYSCALE)
    img2 = cv.imread(reference_image, cv.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Could not open image(s)")
        exit(1)

    sift = cv.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < good_match_precent * n.distance:
            good_matches.append(m)

    if len(good_matches) > max_features:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        height, width = img2.shape
        aligned = cv.warpPerspective(img1, M, (width, height))
        cv.imwrite('saved_images/aligned.png', aligned)

    else:
        print("Not enough matches are found - %d/%d" % (len(good_matches), max_features))
        matchesMask = None

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img3 = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    # cv.imwrite('saved_images/aligned_2.png', img3)
    img3_rgb = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
    plt.imsave('saved_images/matches.png', img3_rgb)

    print(f"Keypoints detected - Reference: {len(kp1)}, To align: {len(kp2)}")
    print(f"Good matches found: {len(good_matches)}")

    plt.imshow(img3_rgb)
    plt.show()


if __name__ == '__main__':
    os.makedirs('saved_images', exist_ok=True)

    harris_corner('reference_img.png')
    sift('align_this.jpg', 'reference_img.png', 10, 0.7)