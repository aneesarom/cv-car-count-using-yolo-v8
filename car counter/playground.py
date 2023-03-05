# Import necessary libraries
import cv2
from ultralytics import YOLO
import cvzone

# Open the video file
cap = cv2.VideoCapture('../Videos/cars.mp4')

# yolo model creation
model = YOLO("../yolo-weights/yolov8n.pt")

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

# Loop through each frame of the video
while True:
    # Read the frame from the video file; success: frame successfully read or not, img: pixel values of the image
    success, img = cap.read()

    # Check if the frame was read successfully
    if not success:
        break

    results = model(img, stream=True)  # generator of Results objects

    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        for box in boxes:
            # box with xyxy format. we are extracting first 4 elements
            x1, y1, x2, y2 = box.xyxy[0]  # remaining 2 element 5: confidence score, 6: className
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # to avoid float values
            cls = int(box.cls[0])
            # we are calculating width and hieght based on upper left(x, y) and lower right(x, y)
            w, h = x2 - x1, y2 - y1
            # highlight or outline objects detected by object detection models.
            cvzone.cornerRect(img, (x1, y1, w, h), l=15, rt=5, colorR=(255, 0, 0))
            cvzone.putTextRect(img, f"{classNames[cls]}", (x1, y1), scale=1.2, offset=10, thickness=2)

    # Display the frame
    cv2.imshow('Video', img)

    # Exit the program if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()  # release the video file
cv2.destroyAllWindows()  # close all windows
