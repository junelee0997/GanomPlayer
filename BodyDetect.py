import cv2
import numpy as np
import torch
def detection(image, show=False):
    image = image[27:587,10:802]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV) #HSV 변경
    col = [104, 88, 14, 157, 73, 26] # 색상을 0~180, 채도와 명도를 0~255로 표현
    order = ['head', 'body', 'leftLeg', 'rightLeg', 'leftArm', 'rightArm']
    position = {order[i] : [0, 0] for i in range(6)}
    area = {order[i] : 0 for i in range(6)}
    cnt = 0
    for i in col:
        lower = np.array([i-3, 80, 80])
        upper = np.array([i+3, 255, 254])

        mask = cv2.inRange(hsv_image, lower, upper)

        points = cv2.findNonZero(mask)
        if points is not None:
            x = np.array(points).T[0]
            y = np.array(points).T[1]
            label1 = (x.max(), y.max())
            label2 = (x.min(), y.min())
            if show: cv2.rectangle(image, label1, label2, (255, 0, 0), 1)
            position[order[cnt]] = [(label1[0] + label2[0]) / 2, (label1[1] + label2[1]) / 2]
            area[order[cnt]] = (label1[0] - label2[0]) * (label1[1] - label2[1])
        else:
            position[order[cnt]] = [0, 0]
            area[order[cnt]] = 0
        cnt += 1

    for i in order:
        if i == 'body': continue
        position[i][0] -= position['body'][0]
        position[i][1] -= position['body'][1]
    if show:
        cv2.imshow("img", image)
        cv2.waitKey()
        cv2.destroyAllWindows()
    return list(position.values()), list(area.values())