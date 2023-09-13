import cv2
import numpy as np
from functions import convert_frame, tune_threshold, video_setup, bbox_contact
from ultralytics import YOLO
import yaml
from functions import tracking_accuracy

with open('config.yaml', 'r') as file:
    yml = yaml.safe_load(file)

video_name = yml['video_name']
botsort_args = yml['botsort']
video_path = f'videos/{video_name}.wmv'

cap = cv2.VideoCapture(video_path)
setup = video_setup(video_name)
cap.set(cv2.CAP_PROP_POS_FRAMES, setup)
ret, frame = cap.read()
segmentation = botsort_args['adaptive_thresholding']
single_class = botsort_args['single_class']
if single_class:
    segmentation = False
    dir1 = 'single_class'
else:
    dir1 = 'contact_class'

if segmentation:
    yolo_model_dir = f'yolo/{dir1}/runs/detect/gray_detection/weights/best.pt'
    output_path = f'results/botsort_tracker/adaptive_thresholding/{video_name}.mp4'
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    worm_threshold = tune_threshold(gray_frame, 101, 12)
    acc_colour = 255
else:
    yolo_model_dir = f'yolo/{dir1}/runs/detect/coloured_detection/weights/best.pt'
    output_path = f'results/botsort_tracker/normal/{dir1}/{video_name}.mp4'
    acc_colour = 0
model = YOLO(yolo_model_dir)
H, W = frame.shape[:2]
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 10.5, (W, H))

id_dict = {}
track_dist = {}
width_dict = {}
height_dict = {}
objects = {}
colors = {}
trajectory = {}
tracker_accuracy = []
vid_frames = 0
contacts = []
male_id = None
while ret:
    frame_count = cap.get(cv2.CAP_PROP_POS_FRAMES) - setup
    if segmentation:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = convert_frame(frame, worm_threshold)
        # Set all three channels to be identical to the grayscale image
        frame = np.stack((frame,) * 3, axis=-1)

    results = model.track(frame, persist=True)[0]
    detections = []
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, track_id, _, class_id = result
        x1 = int(x1)
        x2 = int(x2)
        y1 = int(y1)
        y2 = int(y2)
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        if track_id not in trajectory.keys():
            trajectory[track_id] = [(x_center, y_center)]
        if track_id not in colors.keys():
            colors[track_id] = (255, 0, 0) 
        if track_id not in objects.keys():
            objects[track_id] = results.names[int(class_id)] 
        if vid_frames == 20:
            # calculate the distance between current point and initial point
            x_center0 = trajectory[track_id][0][0]
            y_center0 = trajectory[track_id][0][1]
            dx = abs(x_center - x_center0) ** 2 
            dy = abs(y_center - y_center0) ** 2 
            eud = (dx + dy) ** 0.5
            track_dist[track_id] = eud
        elif vid_frames == 21:
            # get object that has moved the most within the first 5 frames
            max_id = max(track_dist, key=lambda k: track_dist[k])
            if track_id == max_id:
                objects[track_id] = 'male worm'
                colors[track_id] = (0, 0, 255)
                male_id = track_id
            else:
                objects[track_id] = 'female worm'

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), colors[track_id], 2)
        cv2.circle(frame, (x_center, y_center), 3, colors[track_id], -1)
        cv2.putText(frame, objects[track_id], (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    colors[track_id], 1, cv2.LINE_AA)
        
        trajectory[track_id].append((x_center, y_center))
        # Connect the previous centers to the current center with a red line
        for i in range(1, len(trajectory[track_id])):
            prev_center = trajectory[track_id][i - 1]
            current_center = trajectory[track_id][i]
            cv2.line(frame, (int(prev_center[0]), int(prev_center[1])), (int(current_center[0]), int(current_center[1])), colors[track_id], 2)

    # Calculate and display tracking accuracy
    if (dir1 == 'single_class') and (male_id is not None):
        tracking_accuracy(male_id, results.boxes.data.tolist(), tracker_accuracy, objects)
        # Get top right coordinates to display average tracking accuracy
        acc_x = W - 200
        acc_y = 60
        cv2.putText(frame, f'Acc: {round(np.mean(tracker_accuracy), 1)}%', (acc_x, acc_y), cv2.FONT_HERSHEY_SIMPLEX,
                    1, acc_colour, 2, cv2.LINE_AA)

    if vid_frames <= 21:
        vid_frames += 1
    elif (vid_frames == 22) and (single_class is True):
        # Compute euclidean distance between male and female worms
        bbox_contact(trajectory, male_id, objects, frame_count, contacts, frame, 255)

    # Resize and then display frame and write to output video
    display_frame = cv2.resize(frame, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_LINEAR)
    cv2.imshow('C-elegan BotSORT Tracker', display_frame)
    cap_out.write(frame)
    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()

# write time of contact to text file
contact_path = output_path[:-4] + '.txt'  # Replace .mp4 with .txt
contacts = sorted(set(contacts))
contacts = ','.join(contacts).replace(',', '\n')
with open(contact_path, "w") as contacts_out:
    contacts_out.write(contacts)

male_trajectory = trajectory[male_id]
print()
print('END:' male_trajectory)
# write male trajectory to text file
male_trajectory_path = output_path[:-4] + '_trajectory.txt' 
male_trajectory = ','.join([str(trajectories) for trajectories in male_trajectory]).replace('),', ')\n')
with open(male_path, "w") as male_trajectory_path:
    male_out.write(male_trajectory)
