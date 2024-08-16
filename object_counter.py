# from collections import defaultdict
# import cv2
# from shapely.geometry import LineString, Point, Polygon
# from ultralytics.utils.checks import check_imshow, check_requirements
# from ultralytics.utils.plotting import Annotator, colors

# check_requirements("shapely>=2.0.0")


# class ObjectCounter:
#     """A class to manage the counting of objects in a real-time video stream based on their tracks."""

#     def __init__(self):
#         """Initializes the Counter with default values for various tracking and counting parameters."""
#         # Mouse events
#         self.is_drawing = False
#         self.selected_point = None

#         # Region & Line Information
#         self.reg_pts = [(20, 400), (1260, 400)]
#         self.line_dist_thresh = 15
#         self.counting_region = None
#         self.region_color = (255, 0, 255)
#         self.region_thickness = 5

#         # Image and annotation Information
#         self.im0 = None
#         self.tf = None
#         self.view_img = False
#         self.view_in_counts = True
#         self.view_out_counts = True

#         self.names = {}  # Classes names
#         self.annotator = None  # Annotator
#         self.window_name = "AI-Based Barista Productivity Monitoring System"

#         # Object counting Information
#         self.in_counts = 0
#         self.out_counts = 0
#         self.count_ids = []
#         self.class_wise_count = defaultdict(lambda: defaultdict(int))
#         self.count_txt_thickness = 0
#         self.count_txt_color = (255, 255, 255)
#         self.count_bg_color = (255, 255, 255)
#         self.cls_txtdisplay_gap = 50
#         self.fontsize = 0.6

#         # Tracks info
#         self.track_history = defaultdict(list)
#         self.track_thickness = 2
#         self.draw_tracks = False
#         self.track_color = None

#         # Check if environment support imshow
#         self.env_check = check_imshow(warn=True)

#     def set_args(
#         self,
#         classes_names,
#         reg_pts,
#         count_reg_color=(255, 0, 255),
#         count_txt_color=(0, 0, 0),
#         count_bg_color=(255, 255, 255),
#         line_thickness=2,
#         track_thickness=2,
#         view_img=False,
#         view_in_counts=True,
#         view_out_counts=True,
#         draw_tracks=False,
#         track_color=None,
#         region_thickness=5,
#         line_dist_thresh=15,
#         cls_txtdisplay_gap=50,  # Display gap between each class count
#     ):

#         self.tf = line_thickness
#         self.view_img = view_img
#         self.view_in_counts = view_in_counts
#         self.view_out_counts = view_out_counts
#         self.track_thickness = track_thickness
#         self.draw_tracks = draw_tracks

#         # Region and line selection
#         if len(reg_pts) == 2:
#             print(
#                 "AI-Based Barista Productivity Monitoring System Line Counter Initiated."
#             )
#             self.reg_pts = reg_pts
#             self.counting_region = LineString(self.reg_pts)
#         elif len(reg_pts) >= 3:
#             print(
#                 "AI-Based Barista Productivity Monitoring System Polygon Counter Initiated."
#             )
#             self.reg_pts = reg_pts
#             self.counting_region = Polygon(self.reg_pts)
#         else:
#             print(
#                 "Invalid Region points provided, region_points must be 2 for lines or >= 3 for polygons."
#             )
#             print("Using Line Counter Now")
#             self.counting_region = LineString(self.reg_pts)

#         self.names = classes_names
#         self.track_color = track_color
#         self.count_txt_color = count_txt_color
#         self.count_bg_color = count_bg_color
#         self.region_color = count_reg_color
#         self.region_thickness = region_thickness
#         self.line_dist_thresh = line_dist_thresh
#         self.cls_txtdisplay_gap = cls_txtdisplay_gap

#     def mouse_event_for_region(self, event, x, y, flags, params):

#         if event == cv2.EVENT_LBUTTONDOWN:
#             for i, point in enumerate(self.reg_pts):
#                 if (
#                     isinstance(point, (tuple, list))
#                     and len(point) >= 2
#                     and (abs(x - point[0]) < 10 and abs(y - point[1]) < 10)
#                 ):
#                     self.selected_point = i
#                     self.is_drawing = True
#                     break

#         elif event == cv2.EVENT_MOUSEMOVE:
#             if self.is_drawing and self.selected_point is not None:
#                 self.reg_pts[self.selected_point] = (x, y)
#                 self.counting_region = Polygon(self.reg_pts)

#         elif event == cv2.EVENT_LBUTTONUP:
#             self.is_drawing = False
#             self.selected_point = None

#     def extract_and_process_tracks(self, tracks):
#         """Extracts and processes tracks for object counting in a video stream."""

#         # Annotator Init and region drawing
#         self.annotator = Annotator(self.im0, self.tf, self.names)

#         # Draw region or line
#         self.annotator.draw_region(
#             reg_pts=self.reg_pts,
#             color=self.region_color,
#             thickness=self.region_thickness,
#         )

#         if tracks[0].boxes.id is not None:
#             boxes = tracks[0].boxes.xyxy.cpu()
#             clss = tracks[0].boxes.cls.cpu().tolist()
#             track_ids = tracks[0].boxes.id.int().cpu().tolist()

#             # Calculate the centroid of each bounding box
#             centroids = []
#             for box in boxes:
#                 centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
#                 centroids.append(centroid)

#             # Find the centroid that is closest to the polygon
#             nearest_centroid = min(
#                 centroids, key=lambda x: self.counting_region.distance(Point(x))
#             )

#             # Extract tracks
#             for box, track_id, cls in zip(boxes, track_ids, clss):
#                 # Draw bounding box
#                 if self.names[cls] in ["John", "Anna"]:
#                     count_label = f"{self.names[cls]} -> cups {self.class_wise_count[self.names[cls]]['Total']}"
#                 else:
#                     count_label = f"{self.names[cls]}"
#                 self.annotator.box_label(
#                     box,
#                     label=count_label,
#                     color=colors(int(track_id), True),
#                 )

#                 # Store class info
#                 if self.names[cls] not in self.class_wise_count:
#                     self.class_wise_count[self.names[cls]] = defaultdict(int)

#                 # Draw Tracks
#                 track_line = self.track_history[track_id]
#                 track_line.append(
#                     (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
#                 )
#                 if len(track_line) > 30:
#                     track_line.pop(0)

#                 # Draw track trails
#                 if self.draw_tracks:
#                     self.annotator.draw_centroid_and_tracks(
#                         track_line,
#                         color=(
#                             self.track_color
#                             if self.track_color
#                             else colors(int(track_id), True)
#                         ),
#                         track_thickness=self.track_thickness,
#                     )

#                 prev_position = (
#                     self.track_history[track_id][-2]
#                     if len(self.track_history[track_id]) > 1
#                     else None
#                 )

#                 # Count objects in any polygon
#                 if len(self.reg_pts) >= 3 and Point(nearest_centroid).within(
#                     self.counting_region
#                 ):
#                     is_inside = self.counting_region.contains(Point(track_line[-1]))

#                     if (
#                         prev_position is not None
#                         and is_inside
#                         and track_id not in self.count_ids
#                     ):
#                         self.count_ids.append(track_id)

#                         if (box[0] - prev_position[0]) * (
#                             self.counting_region.centroid.x - prev_position[0]
#                         ) > 0:
#                             self.in_counts += 1
#                             self.class_wise_count[self.names[cls]]["IN"] += 1
#                             self.class_wise_count[self.names[cls]]["Total"] += 1
#                             # Update staff count
#                             if self.names[cls] == "John":
#                                 self.class_wise_count["John"]["Total"] += 1
#                             elif self.names[cls] == "Anna":
#                                 self.class_wise_count["Anna"]["Total"] += 1
#                         else:
#                             self.out_counts += 1
#                             self.class_wise_count[self.names[cls]]["OUT"] += 1
#                             self.class_wise_count[self.names[cls]]["Total"] -= 1
#                             # Update staff count
#                             if self.names[cls] == "John":
#                                 self.class_wise_count["John"]["Total"] -= 1
#                             elif self.names[cls] == "Anna":
#                                 self.class_wise_count["Anna"]["Total"] -= 1

#         labels_dict = {}

#         for key, value in self.class_wise_count.items():
#             if value["IN"] != 0 or value["OUT"] != 0:
#                 if not self.view_in_counts and not self.view_out_counts:
#                     continue
#                 elif not self.view_in_counts:
#                     labels_dict[str.capitalize(key)] = f"OUT {value['OUT']}"
#                 elif not self.view_out_counts:

#                     labels_dict[str.capitalize(key)] = f"IN {value['IN']}"
#                 else:
#                     labels_dict[str.capitalize(key)] = (
#                         f"IN {value['IN']} OUT {value['OUT']}"
#                     )

#         if labels_dict is not None:
#             self.annotator.display_analytics(
#                 self.im0, labels_dict, self.count_txt_color, self.count_bg_color, 10
#             )

#         return self.im0

#     def display_table(self, image):
#         """Display a table of the number of cups served by John and Anna."""
#         # Define table position and size
#         table_x = 10
#         table_y = 10
#         table_width = 200
#         table_height = 100

#         # Create a blank table image
#         table_image = cv2.rectangle(
#             image.copy(),
#             (table_x, table_y),
#             (table_x + table_width, table_y + table_height),
#             (255, 255, 255),
#             -1,
#         )
#         cv2.putText(
#             table_image,
#             "John: {}".format(self.class_wise_count["John"]["Total"]),
#             (table_x + 10, table_y + 30),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 0, 0),
#             2,
#         )
#         cv2.putText(
#             table_image,
#             "Anna: {}".format(self.class_wise_count["Anna"]["Total"]),
#             (table_x + 10, table_y + 60),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.6,
#             (0, 0, 0),
#             2,
#         )

#         # Overlay the table image on the original image
#         cv2.addWeighted(table_image, 0.6, image, 0.4, 0, image)

#         return image

#     def display_frames(self):
#         """Display frame."""
#         if self.env_check:
#             cv2.namedWindow(self.window_name)
#             if len(self.reg_pts) == 4:  # only add mouse event If user drawn region
#                 cv2.setMouseCallback(
#                     self.window_name,
#                     self.mouse_event_for_region,
#                     {"region_points": self.reg_pts},
#                 )
#             cv2.imshow(self.window_name, self.im0)
#             # Break Window
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 return

#     def start_counting(self, im0, tracks):
#         """
#         Main function to start the object counting process."""
#         self.im0 = im0  # store image
#         self.extract_and_process_tracks(tracks)  # draw region even if no objects
#         self.im0 = self.display_table(self.im0)  # display table of counts

#         if self.view_img:
#             self.display_frames()
#         return self.im0


# if __name__ == "__main__":
#     ObjectCounter()
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

            # Find the centroid that is closest to the polygon
            nearest_centroid = min(
                centroids, key=lambda x: self.counting_region.distance(Point(x))
            )

            # Extract tracks
            for box, track_id, cls in zip(boxes, track_ids, clss):
                # Draw bounding box
                if self.names[cls] in ["yttsn", "wpptt"]:
                    count_label = f"{self.names[cls]} -> cups {self.class_wise_count[self.names[cls]]['Total']}"
                else:
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

                prev_position = (
                    self.track_history[track_id][-2]
                    if len(self.track_history[track_id]) > 1
                    else None
                )

                if prev_position is not None:
                    if (
                        self.counting_region.contains(Point(prev_position))
                        and self.counting_region.distance(Point(nearest_centroid))
                        <= self.line_dist_thresh
                    ):
                        # Check if this object was already counted
                        if track_id not in self.count_ids:
                            # Count the object as "IN"
                            self.count_ids.append(track_id)
                            self.class_wise_count[self.names[cls]]["IN"] += 1
                            self.class_wise_count[self.names[cls]]["Total"] += 1
                    elif (
                        not self.counting_region.contains(Point(prev_position))
                        and self.counting_region.distance(Point(nearest_centroid))
                        <= self.line_dist_thresh
                    ):
                        # Check if this object was already counted
                        if track_id not in self.count_ids:
                            # Count the object as "OUT"
                            self.count_ids.append(track_id)
                            self.class_wise_count[self.names[cls]]["OUT"] += 1
                            self.class_wise_count[self.names[cls]]["Total"] += 1

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
