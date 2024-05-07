from ultralytics import YOLO
from object_counter import ObjectCounter
import cv2


def main():
    # Load model
    model = YOLO("bestv6.pt")

    # Open video capture
    cap = cv2.VideoCapture("lastyouneed.mp4")

    # for output video
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
            (50, 400),
            (80, 400),
            (80, 1080),
            (50, 1080),
        ],  # Define counting region
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

            tracks = model.track(im0, persist=True, show=False)

            im0 = counter.start_counting(im0, tracks)
            video_writer.write(im0)

            # Check for 'q' key press to stop the application
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
