"""This script improves detection by using image segmentation"""

import cv2
import numpy as np
from functions import convert_frame, tune_threshold, video_setup

# Create a CSRT object tracker
tracker = cv2.legacy.MultiTracker_create()

# The beginning of the video is the setup, so we skip this part by reading the 100th frame of the video.
video_name = 'mating6'

video = cv2.VideoCapture(f"videos/{video_name}.wmv")
setup = video_setup(video_name)
video.set(cv2.CAP_PROP_POS_FRAMES, setup)
_, frame = video.read()
zoom = 0.5
frame = cv2.resize(frame, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)

# Set up video writer
# resize the width and height to make it easier to draw bounding boxes
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) * zoom)
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) * zoom)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# lists to hold initial bounding boxes, colors, and worm type
bbox_list = []
bb_colors = []
worm_type = []
trajectory = {}
first = True
count = 0
segmentation = True
output = 'normal'
if segmentation:
    output = 'adaptive_thresholding'
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Display result of segmentation algorithm and allow users to modify the parameters if need be
    worm_threshold = tune_threshold(gray_frame, 101, 8)
    frame = convert_frame(gray_frame, worm_threshold)
    frame = np.stack((frame,) * 3, axis=-1)

cap_out = cv2.VideoWriter(f'results/csrt/{output}/{video_name}.mp4', fourcc, 10.5, (width, height))

while True:
    # Select worms in the first frame of the video. First worm should be the male worm.
    bbox = cv2.selectROI('CSRT Tracker', frame, False)
    bbox_list.append(bbox)
    # first bounding box is for male worm, so we select a specific color and text to distinguish from female worms
    if first:
        bb_colors.append((255, 0, 0))
        worm_type.append('Male worm')
        first = False
    # We then select a different text and color for all female worms
    else:
        bb_colors.append((0, 0, 255))
        worm_type.append('Female worm')

    # get bbox centers
    center_x = int(bbox[0] + (bbox[2] / 2))
    center_y = int(bbox[1] + (bbox[3] / 2))
    trajectory[count] = [(center_x, center_y)]
    # draw bounding box
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), 255, 2)
    cv2.imshow('CSRT Tracker', frame)
    count += 1
    # Press 'q' to stop selecting worms
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break
# Initialize the CSRT tracker with multiple objects
for bbox in bbox_list:
    tracker.add(cv2.legacy.TrackerCSRT_create(), frame, bbox)

# Loop over the remaining frames in the video
while True:
    # Read the next frame from the video
    success, frame = video.read()
    if not success:
        break
    frame = cv2.resize(frame, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_LINEAR)
    if segmentation:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = convert_frame(frame, worm_threshold)
        frame = np.stack((frame,) * 3, axis=-1)
    success, boxes = tracker.update(frame)
    # If the tracker successfully tracked any objects, draw a bounding box around them
    if success:
        for j, newbox in enumerate(boxes):
            x, y, w, h = [int(k) for k in newbox]
            # Get center of bbox so we can show trajectory
            center_x = int(x + (w / 2))
            center_y = int(y + (h / 2))
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), bb_colors[j], 2)
            cv2.circle(frame, (center_x, center_y), 3, bb_colors[j], -1)
            cv2.putText(frame, worm_type[j], (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bb_colors[j], 2)
            # Update trajectory of the worm
            trajectory[j].append((center_x, center_y))
            for i in range(1, len(trajectory[j])):
                prev_center = trajectory[j][i - 1]
                current_center = trajectory[j][i]
                cv2.line(frame, (int(prev_center[0]), int(prev_center[1])),
                         (int(current_center[0]), int(current_center[1])), bb_colors[j],
                         2)  # Connect the previous center to the current center with a red line
    # Display the resulting frame.
    cv2.imshow('C-elegan CSRT Tracker', frame)
    # Save the frame.
    cap_out.write(frame)
    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video.release()
cap_out.release()
cv2.destroyAllWindows()
