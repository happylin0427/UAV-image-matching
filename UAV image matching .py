
import cv2
import numpy as np
from matplotlib import pyplot as plt
from enum import Enum





def match(img1, img2):



    detector = cv2.ORB_create(54000)

    descriptor = cv2.xfeatures2d.BEBLID_create(0.75)  # ORB/

    e01 = cv2.getTickCount()

    kp1 = detector.detect(img1, None)
    kp2 = detector.detect(img2, None)

    e02 = cv2.getTickCount()

    kp1, des1 = descriptor.compute(img1, kp1)
    kp2, des2 = descriptor.compute(img2, kp2)

    e03 = cv2.getTickCount()

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_all = matcher.match(des1, des2)

    e04 = cv2.getTickCount()

    t11 = (e02 - e01) / cv2.getTickFrequency()
    t12 = (e03 - e02) / cv2.getTickFrequency()
    t13 = (e04 - e03) / cv2.getTickFrequency()
    return matches_all,kp1, kp2,t11,t12,t13



class DrawingType(Enum):
    ONLY_LINES = 1
    LINES_AND_POINTS = 2
    COLOR_CODED_POINTS_X = 3
    COLOR_CODED_POINTS_Y = 4
    COLOR_CODED_POINTS_XpY = 5


def draw_matches(src1, src2, kp1, kp2, matches, drawing_type):
    height = max(src1.shape[0], src2.shape[0])
    width = src1.shape[1] + src2.shape[1]
    output = np.zeros((height, width, 3), dtype=np.uint8)
    output[0:src1.shape[0], 0:src1.shape[1]] = src1
    output[0:src2.shape[0], src1.shape[1]:] = src2[:]

    if drawing_type == DrawingType.ONLY_LINES:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255))

    elif drawing_type == DrawingType.LINES_AND_POINTS:
        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.line(output, tuple(map(int, left)), tuple(map(int, right)), (0, 255, 255),1)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))
            cv2.circle(output, tuple(map(int, left)), 10, (255, 0, 0),-2, 2)
            cv2.circle(output, tuple(map(int, right)), 10, (255, 0, 0),-2, 2)

    elif drawing_type == DrawingType.COLOR_CODED_POINTS_X or drawing_type == DrawingType.COLOR_CODED_POINTS_Y or drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
        _1_255 = np.expand_dims(np.array(range(0, 256), dtype='uint8'), 1)
        _colormap = cv2.applyColorMap(_1_255, cv2.COLORMAP_HSV)

        for i in range(len(matches)):
            left = kp1[matches[i].queryIdx].pt
            right = tuple(sum(x) for x in zip(kp2[matches[i].trainIdx].pt, (src1.shape[1], 0)))

            if drawing_type == DrawingType.COLOR_CODED_POINTS_X:
                colormap_idx = int(left[0] * 256. / src1.shape[1])  # x-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_Y:
                colormap_idx = int(left[1] * 256. / src1.shape[0])  # y-gradient
            if drawing_type == DrawingType.COLOR_CODED_POINTS_XpY:
                colormap_idx = int((left[0] - src1.shape[1] * .5 + left[1] - src1.shape[0] * .5) * 256. / (
                            src1.shape[0] * .5 + src1.shape[1] * .5))  # manhattan gradient

            color = tuple(map(int, _colormap[colormap_idx, 0, :]))
            cv2.circle(output, tuple(map(int, left)), 10, color, -2,2)
            cv2.circle(output, tuple(map(int, right)), 10, color,-2,2)
    return output


MIN_MATCH_COUNT = 10
# 原图


img1 = cv2.imread('./data/DJI_0015.JPG', 0)  # queryImage
img2 = cv2.imread('./data/DJI_0016.JPG', 0)  # trainImage


img11 =cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img22 =cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)


e0 = cv2.getTickCount()
matches_all,kp1, kp2,t11,t12,t13 = match(img1, img2)

e1 = cv2.getTickCount()
matches_gms =cv2.xfeatures2d.matchGMS(img1.shape[:2], img2.shape[:2], kp1, kp2, matches_all, withScale=True, withRotation=True, thresholdFactor=6)#graff
e2= cv2.getTickCount()


if len(matches_gms) > MIN_MATCH_COUNT:
    # 获取关 点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches_gms]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[n.trainIdx].pt for n in matches_gms]).reshape(-1, 1, 2)


    #F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.LMEDS)
    F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.LMEDS)
    #F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC,2.5)
    num = str(mask.tolist()).count("1")
    num1 = str(mask.tolist()).count("0")
    matchesMask = mask.ravel().tolist()

else:
    print("Not enough matches are found - %d/%d" % (len(matches_gms), MIN_MATCH_COUNT))
    matchesMask = None


e3 = cv2.getTickCount()
t1 = (e1 - e0) / cv2.getTickFrequency()
t2 = (e2 - e1) / cv2.getTickFrequency()
t3 = (e3 - e2) / cv2.getTickFrequency()
t = (e3 - e0) / cv2.getTickFrequency()
draw_params = dict(

                   matchColor=(0,255,255),  # draw matches in cyan color
                   singlePointColor=(0,0,255),
                   matchesMask=matchesMask,  # draw only inliers
                   flags=8)


output = cv2.drawMatches(img11, kp1, img22, kp2, matches_gms, mask, **draw_params)

inlier_ratio =float(num)/ float(len(matches_gms))
outlier_ratio =float(num1)/ float(len(matches_gms))




print('Matching Results')
print('*******************************')

print('# Keypoints 1:                        \t', len(kp1))
print('# Keypoints 2:                        \t', len(kp2))
print('# Matchesall:                            \t', len(matches_all))


print('# num:                               \t', (num))
print('# mask:                               \t', len(mask))

print('# Inliers Ratio:                      \t', inlier_ratio)
print('# outliers Ratio:                      \t', outlier_ratio)
print(t11,t12, t13,t1,t2,t3,t)


plt.imshow(output), plt.show()
cv2.waitKey(0)