import cv2
import numpy as np
import time

video = cv2.VideoCapture(0)

time.sleep(2)
background = 0


for i in range(30):
    ret, background= video.read()
background=np.flip(background,axis=1)


while True:
    ret, image=video.read()
    image=np.flip(image, axis=1)
    hsv=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur=cv2.GaussianBlur(hsv,(35,35),0)

    lower=np.array([0,120,70])
    upper=np.array([10,255,255])
    mask01=cv2.inRange(hsv,lower,upper)

    lower_red = np.array([160, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask02 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask01 + mask02

    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

    image[np.where(mask==255)]=background[np.where(mask==255)]

    cv2.imshow("Cload",mask01)
    cv2.imshow("Clod",mask02)
    cv2.imshow("Cloa",image)
    cv2.imshow("Background",background)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break

video.release()
cv2.destroyAllWindows()
