import cv2
import numpy as np

apples_img = cv2.imread('apels.jpg', cv2.IMREAD_UNCHANGED)
apple_img = cv2.imread('apple.jpg', cv2.IMREAD_UNCHANGED)

cv2.imshow('Template',apple_img)

result = cv2.matchTemplate(apples_img, apple_img, cv2.TM_CCOEFF_NORMED)

cv2.imshow('Template Matching',result)

min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

threshold = .40


yloc, xloc = np.where(result >= threshold)

w = apple_img.shape[1]
h = apple_img.shape[0]

rectangles = []
for (x,y) in zip(xloc,yloc):
    rectangles.append([int(x),int(y),int(w),int(h)])
    rectangles.append([int(x),int(y),int(w),int(h)])

rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)
print(len(rectangles))

for (x,y,w,h) in rectangles:
    cv2.rectangle(apples_img,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.putText(apples_img, "APPLE", (x+20,y+25), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,255), 2)
    # cv2.putText(apples_img, result, (x+5,y+25), cv2.FONT_HERSHEY_COMPLEX, 0.65, (0,255,255), 2)

cv2.imshow('Hasil',apples_img)

cv2.waitKey()
cv2.destroyAllWindows()
