from collections import defaultdict
import cv2
from shapely.geometry import LineString, Point, Polygon
from ultralytics.utils.checks import check_imshow, check_requirements
from ultralytics.utils.plotting import Annotator, colors

check_requirements("shapely>=2.0.0")


class ObjectCounter:
    """A class to manage the counting of objects in a real-time video stream based on their tracks."""

    def __init__(self):
        """Initializes the Counter with default values for various tracking and counting parameters."""
        # Mouse events
        self.is_drawing = False
        self.selected_point = None

        # Region & Line Information
        self.reg_pts = [(20, 400), (1260, 400)]
        self.line_dist_thresh = 15
        self.counting_region = None
        self.region_color = (255, 0, 255)
        self.region_thickness = 5

        # Image and annotation Information
        self.im0 = None
        self.tf = None
        self.view_img = False
        self.view_in_counts = True
        self.view_out_counts = True

        self.names = {}  # Classes names
        self.annotator = None  # Annotator
        self.window_name = "AI-Based Barista Productivity Monitoring System"

        # Object counting Information
        self.in_counts = 0
        self.out_counts = 0
        self.count_ids = []
        self.class_wise_count = defaultdict(lambda: defaultdict(int))
        self.count_txt_thickness = 0
        self.count_txt_color = (255, 255, 255)
        self.count_bg_color = (255, 255, 255)
        self.cls_txtdisplay_gap = 50
        self.fontsize = 0.6

        # Tracks info
        self.track_history = defaultdict(list)
        self.track_thickness = 2
        self.draw_tracks = False
        self.track_color = None

        # Check if environment support imshow
        self.env_check = check_imshow(warn=True)

    def set_args(
        self,
        classes_names,
        reg_pts,
        count_reg_color=(255, 0, 255),
        count_txt_color=(0, 0, 0),
        count_bg_color=(255, 255, 255),
        line_thickness=2,
        track_thickness=2,
        view_img=False,
        view_in_counts=True,
        view_out_counts=True,
        draw_tracks=False,
        track_color=None,
        region_thickness=5,
        line_dist_thresh=15,
        cls_txtdisplay_gap=50,  # Display gap between each class count
    ):

        self.tf = line_thickness
        self.view_img = view_img
        self.view_in_counts = view_in_counts
        self.view_out_counts = view_out_counts
        self.track_thickness = track_thickness
        self.draw_tracks = draw_tracks

        # Region and line selection
        if len(reg_pts) == 2:
            print(
                "AI-Based Barista Productivity Monitoring System Line Counter Initiated."
            )
            self.reg_pts = reg_pts
            self.counting_region = LineString(self.reg_pts)
        elif len(reg_pts) >= 3:
            print(
                "AI-Based Barista Productivity Monitoring System Polygon Counter Initiated."
            )
            self.reg_pts = reg_pts
            self.counting_region = Polygon(self.reg_pts)
        else:
            print(
                "Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons."
            )
            print("Using Line Counter Now")
            self.counting_region = LineString(self.reg_pts)

        self.names = classes_names
        self.track_color = track_color
        self.count_txt_color = count_txt_color
        self.count_bg_color = count_bg_color
        self.region_color = count_reg_color
        self.region_thickness = region_thickness
        self.line_dist_thresh = line_dist_thresh
        self.cls_txtdisplay_gap = cls_txtdisplay_gap

    def mouse_event_for_region(self, event, x, y, flags, params):

        if event == cv2.EVENT_LBUTTONDOWN:
            for i, point in enumerate(self.reg_pts):
                if (
                    isinstance(point, (tuple, list))
                    and len(point) >= 2
                    and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
                ):
                    self.selected_point = i
                    self.is_drawing = True
                    break

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.is_drawing and self.selected_point is not None:
                self.reg_pts[self.selected_point] = (x, y)
                if len(self.reg_pts) == 2:
                    self.counting_region = LineString(self.reg_pts)
                else:
                    self.counting_region = Polygon(self.reg_pts)

        elif event == cv2.EVENT_LBUTTONUP:
            self.is_drawing = False
            self.selected_point = None

    def extract_and_process_tracks(self, tracks):
        """Extracts and processes tracks for object counting in a video stream."""

        # Annotator Init and region drawing
        self.annotator = Annotator(self.im0, self.tf, self.names)

        # Draw region or line
        self.annotator.draw_region(
            reg_pts=self.reg_pts,
            color=self.region_color,
            thickness=self.region_thickness,
        )

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu()
            clss = tracks[0].boxes.cls.cpu().tolist()
            track_ids = tracks[0].boxes.id.int().cpu().tolist()

            # Calculate the centroid of each bounding box
            centroids = []
            for box in boxes:
                centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                centroids.append(centroid)

            # Extract tracks
            person_centroids = {}
            cup_centroids = []

            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                if self.names[cls] in ["yttsn", "wpptt"]:
                    person_centroids[track_id] = (
                        (box[0] + box[2]) / 2,
                        (box[1] + box[3]) / 2,
                    )
                    count_label = f"{self.names[cls]} -> cups {self.class_wise_count[self.names[cls]]['Total']}"
                elif self.names[cls] == "cup":
                    cup_centroids.append(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
                    count_label = f"{self.names[cls]}"

                self.annotator.box_label(
                    box,
                    label=count_label,
                    color=colors(int(track_id), True),
                )

                # Store class info
                if self.names[cls] not in self.class_wise_count:
                    self.class_wise_count[self.names[cls]] = defaultdict(int)

                # Draw Tracks
                track_line = self.track_history[track_id]
                track_line.append(
                    (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
                )
                if len(track_line) > 30:
                    track_line.pop(0)

                # Draw track trails
                if self.draw_tracks:
                    self.annotator.draw_centroid_and_tracks(
                        track_line,
                        color=(
                            self.track_color
                            if self.track_color
                            else colors(int(track_id), True)
                        ),
                        track_thickness=self.track_thickness,
                    )

            # Reset count_ids for this frame
            self.count_ids = []

            # For each cup, find the closest person and increment their count if the cup is in the region
            for cup_centroid in cup_centroids:
                if self.counting_region.contains(Point(cup_centroid)):
                    nearest_person = None
                    min_distance = float("inf")
                    for person_id, person_centroid in person_centroids.items():
                        distance = Point(cup_centroid).distance(Point(person_centroid))
                        if distance < min_distance:
                            min_distance = distance
                            nearest_person = person_id

                    if nearest_person is not None:
                        if nearest_person not in self.count_ids:
                            self.count_ids.append(nearest_person)
                            person_class = [
                                cls
                                for track_id, cls in zip(track_ids, clss)
                                if track_id == nearest_person
                            ][0]
                            self.class_wise_count[self.names[person_class]]["IN"] += 1
                            self.class_wise_count[self.names[person_class]][
                                "Total"
                            ] += 1

        return self.annotator.result()

    def start_counting(self, im0, tracks):
        self.im0 = im0
        self.im0 = self.extract_and_process_tracks(tracks)

        if self.env_check and self.view_img:
            # View Image
            cv2.imshow(self.window_name, self.im0)
            cv2.setMouseCallback(self.window_name, self.mouse_event_for_region)

        # Write Class-wise Count on the video
        line_height = 50
        y_offset = 50
        x_offset = 50

        # Draw counts for "yttsn"
        count_txt = f"yttsn Cups Served: {self.class_wise_count['yttsn']['Total']}"
        cv2.putText(
            self.im0,
            count_txt,
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.fontsize,
            self.count_txt_color,
            self.tf,
        )

        y_offset += line_height

        # Draw counts for "wpptt"
        count_txt = f"wpptt Cups Served: {self.class_wise_count['wpptt']['Total']}"
        cv2.putText(
            self.im0,
            count_txt,
            (x_offset, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.fontsize,
            self.count_txt_color,
            self.tf,
        )
        return self.im0
