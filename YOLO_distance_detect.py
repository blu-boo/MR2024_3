import datetime
import cv2
from ultralytics import YOLO
import math

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

# Load class names from file
with open('classes.txt', 'r') as coco128:
    class_list = coco128.read().split('\n')

# Initialize the model
model = YOLO('yolov8m.pt')

# Capture from camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Get camera frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate camera center
camera_center_x = frame_width // 2
camera_center_y = frame_height // 2

try:
    while True:
        # Check if 'q' is pressed early in the loop
        if cv2.waitKey(10) == ord('q'):
            print("Exiting...")
            break

        start = datetime.datetime.now()

        ret, frame = cap.read()
        if not ret:
            print('Cam Error')
            break

        detection = model(frame)[0]

        for data in detection.boxes.data.tolist():  # data : [xmin, ymin, xmax, ymax, confidence_score, class_id]
            confidence = float(data[4])
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            xmin, ymin, xmax, ymax = int(data[0]), int(data[1]), int(data[2]), int(data[3])
            label = int(data[5])
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, 2)
            cv2.putText(frame, class_list[label]+' '+str(round(confidence, 2)) + '%', (xmin, ymin), cv2.FONT_ITALIC, 1, WHITE, 2)

            # Calculate bounding box center
            bbox_center_x = (xmin + xmax) // 2
            bbox_center_y = (ymin + ymax) // 2

            # Draw the center of the bounding box
            cv2.circle(frame, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)

            # Calculate distance from camera center
            distance = math.sqrt((bbox_center_x - camera_center_x) ** 2 + (bbox_center_y - camera_center_y) ** 2)

            # Determine relative position (left/right, up/down)
            position_x = 'right' if bbox_center_x > camera_center_x else 'left'
            position_y = 'down' if bbox_center_y > camera_center_y else 'up'

            # Display distance and position on the frame
            text = f'Distance: {int(distance)} px, Pos: {position_x}, {position_y}'
            cv2.putText(frame, text, (xmin, ymax + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 2)

            # Print the distance and position to the console
            print(f"Bounding Box Center: ({bbox_center_x}, {bbox_center_y})")
            print(f"Distance from Camera Center: {int(distance)} px")
            print(f"Relative Position: {position_x}, {position_y}")
            print('-' * 50)

        end = datetime.datetime.now()

        total = (end - start).total_seconds()
        print(f'Time to process 1 frame: {total * 1000:.0f} milliseconds')

        fps = f'FPS: {1 / total:.2f}'
        cv2.putText(frame, fps, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow('frame', frame)

finally:
    # Ensure resources are released properly
    if cap.isOpened():
        cap.release()
    cv2.destroyAllWindows()
    print("Camera and windows successfully released.")
