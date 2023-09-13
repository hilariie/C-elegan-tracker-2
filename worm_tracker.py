import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import SamPredictor, sam_model_registry
import functions as ft
import yaml

with open('config.yaml', 'r') as file:
    yml = yaml.safe_load(file)

video_name = yml['video_name']
worm_tracker_args = yml['worm_tracker']
video_path = f'videos/{video_name}.wmv'

# Read video
cap = cv2.VideoCapture(video_path)
# skip the part where they set up the worms
setup = ft.video_setup(video_name)
cap.set(cv2.CAP_PROP_POS_FRAMES, setup)
ret, frame = cap.read()
# set variables to be used in video processing
tracker = ft.Tracker()
track_dist = {}
objects = {}
colors = {}
trajectory = {}
contacts = []
male_id = None
tracker_accuracy = []
font = cv2.FONT_HERSHEY_SIMPLEX
line = cv2.LINE_AA

worm_count = 0
n_init = 10
worm_check = yml['modify']
image_segmentation = worm_tracker_args['adaptive_thresholding']
SAM = worm_tracker_args['sam']
single_class = worm_tracker_args['single_class']
if single_class:
    yolo_dir1 = 'single_class'
    image_segmentation = False
else:
    yolo_dir1 = 'contact_class'
yolo_model_dir = f'yolo/{yolo_dir1}/runs/detect/coloured_detection/weights/best.pt'
if image_segmentation:
    output = 'adaptive_thresholding'
    yolo_model_dir = 'yolo/contact_class/runs/detect/gray_detection/weights/best.pt'
    contact_colour = (0, 0, 255)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    worm_threshold = ft.tune_threshold(gray_frame, 101, 12)
elif SAM:
    output = 'sam'
    sam = sam_model_registry["vit_b"](checkpoint="sam_models/model_b.pth").to(device=0)
    mask_predictor = SamPredictor(sam)
    contact_colour = (0, 0, 255)
else:
    output = 'normal'
    contact_colour = 0
model = YOLO(yolo_model_dir)
output_path = f"results/{yolo_dir1}/{output}/{video_name}.mp4"


# set output writer
H, W = frame.shape[:2]
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10.5, (W, H))

while ret:
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES) - setup
    if image_segmentation:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = ft.convert_frame(frame, worm_threshold)
        # Set all three channels to be identical to the grayscale image
        frame = np.stack((frame,) * 3, axis=-1)

    # get predictions
    results = model(frame)[0]
    # list to store predicted bbox and scores of each worm
    detections = []
    contact_bboxes = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        label = results.names[int(class_id)]
        # if contact is predicted with a good confidence, plot bbox and move to next iteration
        if label != 'worm':
            if score > 0.78:
                contact_bboxes.append((x1, y1, x2, y2))
                time_elapsed = frame_count / 10.5
                minutes = int(time_elapsed / 60)
                seconds = int(time_elapsed % 60)
                contacts.append(f"{minutes:02d}:{seconds:02d}")
            continue

        detections.append([x1, y1, x2, y2, score, class_id])
    # check if worm_check is True, so we do not perform this operation all the time.
    if worm_check:
        worm_count, n_init, worm_check = ft.worm_checker(len(detections), worm_count, n_init, worm_check)
    if SAM:
        frame = ft.sam_segmentation(detections, frame, mask_predictor)
    for bbox in contact_bboxes:
        cv2.rectangle(frame, (int(bbox[0]) - 5, int(bbox[1]) - 5), (int(bbox[2]) + 5, int(bbox[3]) + 5),
                      contact_colour, 2)
        cv2.putText(frame, 'CONTACT', (int(bbox[0]) - 5, int(bbox[1]) - 8), font, 0.5,
                    contact_colour, 1, line)
    tracker.update(frame, detections, n_init)

    male_id = ft.draw_tracks_and_trajectories(tracker, trajectory, frame, colors, objects, frame_count, results.names,
                                              male_id, track_dist)

    # Calculate and display tracking accuracy
    if (single_class is True) and (male_id is not None):
        ft.tracking_accuracy(male_id, tracker.tracks, tracker_accuracy, objects)
        # Get top right coordinates to display average tracking accuracy
        acc_x = W - 200
        acc_y = 60
        cv2.putText(frame, f'Acc: {round(np.mean(tracker_accuracy), 1)}%', (acc_x, acc_y), font, 1, contact_colour, 2,
                    line)

    if (frame_count > 21) and (single_class is True):
        # Compute euclidean distance between male and female worms
        ft.bbox_contact(trajectory, male_id, objects, frame_count, contacts, frame, contact_colour)

    # Resize and then display frame and write to output video
    display_frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('C-elegan DeepSORT Tracker', display_frame)
    cap_out.write(frame)
    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

male_trajectory = trajectory[male_id]
# write male trajectory to text file
male_path = f"results/{yolo_dir1}/{output}/{video_name}_trajectory.txt"
male_trajectory = ','.join([str(trajectories) for trajectories in male_trajectory]).replace('),', ')\n')
with open(male_path, "w") as male_out:
    male_out.write(male_trajectory)


# write time of contact to text file
contact_path = f"results/{yolo_dir1}/{output}/{video_name}.txt"
contacts = sorted(set(contacts))
contacts = ','.join(contacts).replace(',', '\n')
with open(contact_path, "w") as contacts_out:
    contacts_out.write(contacts)
