import cv2 as cv
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

object_detector = cv.createBackgroundSubtractorMOG2(history=1000, varThreshold=80)
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Extract region of interest
    roi = frame[290: 480, 30: 1080]       

    mask = object_detector.apply(frame)
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        if area > 10:
            cv.drawContours(frame, [cnt], -1, (0,255, 0), 2)
            x, y, w, h = cv.boundingRect(cnt)
            cv.rectangle(roi, (x,y),( x + w, y+h), (0,255,0), 3)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow("roi", roi)
    cv.imshow('frame', frame)
    cv.imshow("Mask", mask)
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()