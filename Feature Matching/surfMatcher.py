import cv2 as cv
import matplotlib.pyplot as plt

MIN_MATCH_COUNT = 10
img1 = cv.imread('church1.jpg', 0)  # queryImage
# gray_img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2 = cv.imread('church2.jpg', 0)  # trainImage
# gray_img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# Initiate SIFT detector
surf = cv.xfeatures2d.SURF_create(400)

# find the keypoints and descriptors with SIFT
kp1, des1 = surf.detectAndCompute(img1, None)
kp2, des2 = surf.detectAndCompute(img2, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
# store all the good matches as per Lowe's ratio test.
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)
# if len(good) > MIN_MATCH_COUNT:
#     src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
#     dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
#     M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
#     matchesMask = mask.ravel().tolist()
#     h, w, d = img1.shape
#     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
#     dst = cv.perspectiveTransform(pts, M)
#     img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
# else:
#     print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
#     matchesMask = None
draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                   singlePointColor=None,
                   flags=2)
img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
plt.imshow(img3, 'gray'), plt.show()
