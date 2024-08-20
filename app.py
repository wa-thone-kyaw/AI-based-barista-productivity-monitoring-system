from ultralytics import YOLO
from object_counter import ObjectCounter  # Import ObjectCounter class
import cv2
import torch


def main():
    # Load model
    model = YOLO("besthtila2.pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Open video capture
    cap = cv2.VideoCapture("mm-pjtest.mp4")

    # For output video
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    # Check if video opened successfully
    if not cap.isOpened():
        print("Error opening video file.")
        return

    # Video writer for output
    video_writer = cv2.VideoWriter(
        "Ai_based_barista_productivity_monitoring_system_output.avi",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    # Init Object Counter
    counter = ObjectCounter()
    counter.set_args(
        view_img=True,  # Optional: Set to False to disable frame display
        reg_pts=[
            (50, int(0.5 * h)),
            (80, int(0.5 * h)),
            (80, h),
            (50, h),
        ],
        classes_names=model.names,
        draw_tracks=True,
        line_thickness=2,
    )

    try:
        while cap.isOpened():
            success, im0 = cap.read()
            if not success:
                print(
                    "Video frame is empty or video processing has been successfully completed."
                )
                break

            # Use model tracking
            tracks = model.track(im0, persist=True, show=False)

            # Process the frame using ObjectCounter
            im0 = counter.start_counting(im0, tracks)

            # Write the processed frame to the output video
            video_writer.write(im0)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        cap.release()
        video_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
