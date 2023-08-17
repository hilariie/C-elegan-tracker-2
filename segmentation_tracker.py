import cv2
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
fps = cap.get(cv2.CAP_PROP_FPS)

# Skip setup frames
setup = ft.video_setup(video_name)
cap.set(cv2.CAP_PROP_POS_FRAMES, setup)

# Read the first frame
ret, frame = cap.read()

# Initialize variables for tracking
tracker = ft.Tracker()
track_dist = {}
objects = {}
colors = {}
trajectory = {}
contacts = []
worm_count = 0
n_init = 10
SAM = seg_args['sam']
worm_check = True

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
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H))

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
        cv2.putText(frame, 'CONTACT', (min(minx)-20, min(miny)-23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, contact_colour, 1,
                    cv2.LINE_AA)

    # Update tracker
    tracker.update(frame, detections, n_init)

    _ = ft.draw_tracks_and_trajectories(tracker, trajectory, frame, colors, objects, frame_count, results.names, None,
                                        track_dist)

    # Display the frame and write to output video
    cv2.imshow('C-elegan Tracker', frame)
    cap_out.write(frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    ret, frame = cap.read()

# Release video objects and destroy windows
cap.release()
cap_out.release()
cv2.destroyAllWindows()


# write time of contact to text file
contact_path = f"results/segmentation/{output_}/{video_name}.txt"
contacts = sorted(set(contacts))
contacts = ','.join(contacts).replace(',', '\n')
with open(contact_path, "w") as contacts_out:
    contacts_out.write(contacts)
