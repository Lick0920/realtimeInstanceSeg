import cv2
import numpy as np

# 读取图像
img = cv2.imread('E:\\dataset\\roothair\\train_ori\\000000000001.jpg')
# 创建一个窗口
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
# 调整窗口大小
cv2.resizeWindow('Image', 4908, 7368)
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 显示灰度图像# 创建一个窗口
cv2.namedWindow('Gray Image', cv2.WINDOW_NORMAL)
# 调整窗口大小
cv2.resizeWindow('Gray Image', 4908, 7368)
cv2.imshow('Gray Image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
# 计算梯度
gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, np.ones((3,3),np.uint8))

# 二值化图像
_, thresh = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 膨胀操作
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# 距离变换
dist_transform = cv2.distanceTransform(morph, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)

# 获得未知区域
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(morph, sure_fg)

# 标记连通区域
_, markers = cv2.connectedComponents(sure_fg)

# 增加标记
markers = markers+1

# 在未知区域中设置标记为0
markers[unknown==255] = 0

# 分水岭算法分割图像
markers = cv2.watershed(img, markers)
img[markers == -1] = [0,0,255] # 可视化

# 显示结果
cv2.namedWindow('Segmented Image', cv2.WINDOW_NORMAL)
# 调整窗口大小
cv2.resizeWindow('Segmented Image', 4908, 7368)
cv2.imshow('Segmented Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()