from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
from sort import *

cap = cv2.VideoCapture("../Videos/cars.mp4")  # location for video file

# yolo model creation
model = YOLO("../yolo-weights/yolov8m.pt")

# coco dataset
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Dimension created for counting line
limits = [400, 297, 673, 297]
mask = cv2.imread("mask.png")
# iou_threshold = checking bounding box detection on each frame
# min hits = max detection req to consider
# max_age = maximum number of frames an object is allowed to disappear from the detection region
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

totalCount = []

while True:
    success, img = cap.read()
    # merge mask and video img to detect only in required area
    imgRegion = cv2.bitwise_and(img, mask)
    # reading the image file that we are going to overlay
    imgGraphics = cv2.imread("graphics.png", cv2.IMREAD_UNCHANGED)
    # overlay the image: img, imgGraphics at location (0, 0)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))
    # passing merged frame image to the yolo model
    results = model(imgRegion, stream=True)
    detection = np.empty([0, 5])  # it is like an empty list with zero values
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # calculating width and height
            w, h = x2 - x1, y2 - y1
            # converting cls value to int, so that we could index the list
            cls = int(box.cls[0])
            # Formatting confidence value
            conf = math.ceil((box.conf[0] * 100)) / 100
            # allowing cars and only with confidence above 0.6
            if classNames[cls] == "car" and conf > 0.6:
                currentArray = np.array([x1, y1, x2, y2, conf])  # we  are passing this array to detection
                # it replaces the empty array and stack currentArray vertically
                detection = np.vstack((detection, currentArray))

    # passing detection array to tracker to get id
    resultsTracker = tracker.update(detection)

    # Draws a line using opencv
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 2)

    for result in resultsTracker:
        # resultant value has id of the cars
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2, Id = int(x1), int(y1), int(x2), int(y2), int(Id)
        w, h = x2 - x1, y2 - y1
        # boundary box creation
        cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
        # create text field
        cvzone.putTextRect(img, f"{Id}", (x1, y1), scale=1.2, offset=10, thickness=2)
        # calculating the center point
        cx, cy = x1 + (w // 2), y1 + (h // 2)
        # draws a center point, 5 is radius size
        cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)

        # maximum boundary area to detect the car passed the line or not
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 45:
            # check whether id added to the totalCount or not. if not add and count the no of car passed
            if totalCount.count(Id) == 0:
                totalCount.append(Id)
                cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 2)

    # add total count with the overlay img and here (255, 100) is location, 5 is font size, 5 is thickness
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 5)

    # rendering the image
    cv2.imshow('Video', img)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the 'p' key is pressed, pause the video
    if key == ord('p'):
        cv2.waitKey(-1)  # Wait indefinitely until a key is pressed

    # If the 'q' key is pressed, exit the loop
    elif key == ord('q'):
        break
