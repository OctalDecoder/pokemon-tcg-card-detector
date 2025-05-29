import cv2
import numpy as np
from matplotlib import pyplot as plt

def featureMatchingHomography(template_path, img_path):
    # load images
    template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    img_rgb = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    # SIFT detect + compute
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(template, None)
    kp2, des2 = sift.detectAndCompute(img_gray, None)

    # FLANN matcher
    FLANN_INDEX_KDTREE = 1
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=FLANN_INDEX_KDTREE, trees=5),
        dict(checks=50)
    )
    matches = flann.knnMatch(des1, des2, k=2)

    # ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    MIN_MATCHES = 500
    if len(good) < MIN_MATCHES:
        print(f"Not enough matches: {len(good)} found, {MIN_MATCHES} required")
        return

    print(len(good))
    # build source & destination points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # compute homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # draw the detected template boundary in the scene image
    h, w = template.shape
    corners = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1,1,2)
    warped_corners = cv2.perspectiveTransform(corners, M)
    img_with_border = img_rgb.copy()
    cv2.polylines(img_with_border, [np.int32(warped_corners)], True, (0,255,0), 3, cv2.LINE_AA)

    # draw matches
    draw_params = dict(
        matchColor=(0,255,0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    matched_vis = cv2.drawMatches(
        template, kp1,
        img_with_border, kp2,
        good, None,
        **draw_params
    )

    # display
    plt.figure(figsize=(12,6))
    plt.imshow(cv2.cvtColor(matched_vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


# example invocation
featureMatchingHomography("A1 1.png", "1.png")
