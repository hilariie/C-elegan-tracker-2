import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import functions as ft
import yaml

# Load configuration from config.yaml
with open('config.yaml', 'r') as file:
    yml = yaml.safe_load(file)

# Load video name and segmentation arguments from the configuration
video_name = yml['video_name']
seg_args = yml['segmentation']

# Initialize YOLO model
model = YOLO('yolo/segmentation/runs/segment/train4/weights/best.pt')
video_path = f"videos/{video_name}.wmv"

# Load video
cap = cv2.VideoCapture(video_path)

# Skip setup frames
setup = ft.video_setup(video_name)
cap.set(cv2.CAP_PROP_POS_FRAMES, setup)

# Read the first frame
ret, frame = cap.read()

# Initialize variables for tracking
tracker = ft.Tracker()
tracker_accuracy = []
track_dist = {}
objects = {}
colors = {}
trajectory = {}
contacts = []
worm_count = 0
n_init = 10
SAM = seg_args['sam']
worm_check = yml['modify']
male_id = None
font = cv2.FONT_HERSHEY_SIMPLEX
line = cv2.LINE_AA

# Configure segmentation based on SAM configuration
if SAM:
    output_ = 'sam'
    sam = sam_model_registry["vit_b"](checkpoint="sam_models/model_b.pth").to(device=0)
    mask_predictor = SamPredictor(sam)
    contact_colour = (0, 0, 255)
else:
    output_ = 'normal'
    contact_colour = 0

# Define output video path
output_path = f"results/segmentation/{output_}/{video_name}.mp4"
H, W = frame.shape[:2]

cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10.5, (W, H))

# Process frames
while ret:
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES) - setup
    results = model(frame)[0]
    segment_coord = []
    detections = []
    for worm_result in range(len(results)):
        # Parse bounding box results
        bbox_result = results.boxes.data.tolist()[worm_result]
        bboxx1, bboxy1, bboxx2, bboxy2 = map(int, bbox_result[:4])
        score, class_id = bbox_result[4:]

        detections.append([bboxx1, bboxy1, bboxx2, bboxy2, score, class_id])

        # Get segmentation coordinates
        segment_coord.append(results.masks.xy[worm_result])

    # Check worm if all 6 worms have been identified consecutively
    if worm_check:
        worm_count, n_init, worm_check = ft.worm_checker(len(detections), worm_count, n_init, worm_check)

    # Perform image segmentation if SAM is used
    if SAM:
        frame = ft.sam_segmentation(detections, frame, mask_predictor)

    # Process segmentation coordinates
    new_segment_coord = []
    for coords in segment_coord:
        coord_list = []
        for i in range(1, len(coords)):
            x1, y1 = map(int, coords[i - 1])
            x2, y2 = map(int, coords[i])
            coord_list.append((x1, y1))
            cv2.line(frame, (x1, y1), (x2, y2), (250, 180, 180), 2)
        cv2.line(frame, (x2, y2), (int(coords[0][0]), int(coords[0][1])), (250, 180, 180), 2)
        new_segment_coord.append(coord_list)

    # Detect contacts and highlight it
    minx, miny, maxx, maxy = ft.segment_contact_detection(new_segment_coord)
    if minx:
        time_elapsed = frame_count / 10.5
        minutes = int(time_elapsed / 60)
        seconds = int(time_elapsed % 60)
        contacts.append(f"{minutes:02d}:{seconds:02d}")
        cv2.rectangle(frame, (min(minx)-20, min(miny)-20), (max(maxx)+20, max(maxy)+20), contact_colour, 3)
        cv2.putText(frame, 'CONTACT', (min(minx)-20, min(miny)-23), font, 0.5, contact_colour, 1,
                    line)

    # Update tracker
    tracker.update(frame, detections, n_init)
    male_id = ft.draw_tracks_and_trajectories(tracker, trajectory, frame, colors, objects, frame_count, results.names,
                                              male_id, track_dist)

    # Calculate and display tracking accuracy
    if male_id:
        ft.tracking_accuracy(male_id, tracker.tracks, tracker_accuracy, objects)
        # Get top right coordinates to display average tracking accuracy
        acc_x = W - 200
        acc_y = 60
        cv2.putText(frame, f'Acc: {round(np.mean(tracker_accuracy), 1)}%', (acc_x, acc_y), font, 1, contact_colour, 2,
                    line)

    # Resize and then display frame and write to output video
    display_frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('C-elegan DeepSORT Segmentation Tracker', display_frame)
    cap_out.write(frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

# Release video objects and destroy windows
cap.release()
cap_out.release()
cv2.destroyAllWindows()

male_trajectory = trajectory[male_id]
# write male trajectory to text file
male_path = f"results/segmentation/{output_}/{video_name}_trajectory.txt"
male_trajectory = ','.join([str(trajectories) for trajectories in male_trajectory]).replace('),', ')\n')
with open(male_path, "w") as male_out:
    male_out.write(male_trajectory)

# write time of contact to text file
contact_path = f"results/segmentation/{output_}/{video_name}.txt"
contacts = sorted(set(contacts))
contacts = ','.join(contacts).replace(',', '\n')
with open(contact_path, "w") as contacts_out:
    contacts_out.write(contacts)
