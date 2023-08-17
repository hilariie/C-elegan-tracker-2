<div id="header" align="center">
  <h1>
    C-elegans Tracker
    <img src="results\worm_tracker.jpg" alt="worm tracker screenshot" width="300" align="center"/>
  </h1>
</div>

This project aims to detect and track a male C-elegans worm's mating behaviour amongst female worms. It uses Yolov8 by 
Ultralytics to detect the worms, use CSRT, BotSORT, and DeepSort algorithms to track the worms, detect when contact 
between a male and female worm has been made, and records the contact time for scientific research.

# About the Project
This project compares the performance of different tracking algorithms as mentioned above. In order to use the BotSORT and DeepSORT trackers, a detection algorithm
was needed. YOLOv8 was used in this study for the detection of worms. Several annotation approaches were used and compared.

### Detection
To access the yolo training process, navigate to `yolo` directory where you'd immediately meet directories of the different annotation approaches used each having
the script used to train the model, and the model weights.

### Tracking
1. The Channel and Spatial Reliability Tracking (CSRT) from OpenCV was implemented and used in the `csrt.py` script. To
 execute the script, simply run `python csrt.py` in the directory. To view the results produced from this script (after execution),
navigate to `results/csrt`.
2. The BotSORT tracker which is the default yolov8 tracker was used in the `botsort.py` script. To execute, feel free to
 modify the `botsort` section in the [config.yaml](config.yaml) file which provides the option to apply adaptive thresholding (image segmentation) 
to video frames and use single or contact class yolo model. Run `python botsort.py` in the home directory. To
 view the results produced from this script (after execution), navigate to `results/botsort_tracker`
3. The DeepSORT tracker with bounding boxes was used in the `worm_tracker.py` script. This script can be modified in the
`config.yaml` file to apply image segmentation using Meta AI's Segment Anything Model (SAM) or adaptive thresholding. The script also provides the option to use
'single' or 'contact_class' Yolo models. To view the results from this script (after execution), navigate to `results/single_class` or `results/contact_class`.
4. The DeepSORT tracker with segmentation coordinates alongside bounding boxes was used in the `segmentation_tracker.py` script. This script can be modified using the
`config.yaml` file to apply image segmentation using the SAM model. To view the results from this script (after execution), navigate to `results/segmentation`.

## Dependencies
Due to clashing dependencies, some scripts have differing requirements.
1. `csrt.py`: To run I used a separate virtual environment and pip installed  opencv-contrib-python only.
2. `worm_tracker.py` and `segmentation_tracker.py`: To run you need to install the following:
   - `ultralytics`
   - `TensorFlow`
   - `scikit-image`
   - `filterpy`
   - `numpy`
3. `botsort.py`: To run this script, you'd need the above libraries excluding TensorFlow, and you'd need to use Python v3.10.

I created virtual environments for each of these 3 groups of scripts mentioned above. 

## How to Use
1. Clone the repository using the command: `git clone <repo_url>`
2. Navigate to the project directory: `cd <repo>`
3. Download the videos used for tracking. Unzip and move the videos into the `videos` directory.
4. Modify the `config.yaml` file accordingly to carry out experimentations.

#### For adaptive thresholding
1. if adaptive thresholding is set to True in the `config.yaml` file, after running the script, a grayscale image will appear asking if the segmentation performed on the image is satisfactory.
2. Hit any key and on the terminal, you will be asked if you wish to adjust the parameters to make the segmentation better. This is an iterative process until you are content with the segmentation.

After this, the script will display the output frames of the video with the tracked worms. If you want to end the script/video, press 'q'.
