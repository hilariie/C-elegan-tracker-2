import json


def convert_to_yolov5(json_data):

    images = json_data['images']
    annotations = json_data['annotations']

    for image in images:
        image_id = image['id']
        file_name = image['file_name']
        width = image['width']
        height = image['height']

        # Create annotation file for the image
        annotation_file = file_name.replace('.jpg', '.txt')
        annotation_path = f'img/valid/labels/{annotation_file}'
        with open(annotation_path, 'w') as f:
            for annotation in annotations:
                if annotation['image_id'] == image_id:
                    segmentation = annotation['segmentation']

                    # Flatten the segmentation coordinates
                    flattened_segmentation = [coord for segment in segmentation for coord in segment]

                    # Normalize the coordinates
                    normalized_segmentation = [
                        str(coord / width) if i % 2 == 0 else str(coord / height)
                        for i, coord in enumerate(flattened_segmentation)
                    ]

                    # Create the line with the class_id followed by the normalized coordinates
                    line = f"0 {' '.join(normalized_segmentation)}\n"
                    f.write(line)

        # Append image path to train.txt file
        #with open('train.txt', 'a') as train_file:
        #    train_file.write(file_name + '\n')

    print("Conversion completed successfully!")


data = r"img\valid\_annotations.coco.json"

with open(data, 'r') as json_file:
    j_data = json.load(json_file)

# Convert to YOLOv5 format
convert_to_yolov5(j_data)
