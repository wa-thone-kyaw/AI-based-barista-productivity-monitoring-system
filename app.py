# app.py

from ultralytics import YOLO
from object_counter import ObjectCounter
import cv2

model = YOLO("yolov8m.pt")
cap = cv2.VideoCapture("myouneed.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer
video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080)
)

# Init Object Counter
counter = ObjectCounter()
counter.set_args(
    view_img=True,
    reg_pts=[(50, 400), (80, 400), (80, 1080), (50, 1080)],
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    tracks = model.track(im0, persist=True, show=False)

    im0 = counter.start_counting(im0, tracks)
    video_writer.write(im0)

cap.release()
video_writer.release()
cv2.destroyAllWindows()
