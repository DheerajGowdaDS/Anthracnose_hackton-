'''from ultralytics import YOLO

model=YOLO("best (2).pt")

results=model.predict(r'D:\pycharm_projects\leaf\video_frames1\frame_358.jpg',show=True, conf=0.25, save=True)'''

import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("best (2).pt")

# Define input and output folders
input_folder = r'D:\pycharm_projects\leaf\video_frames1'
output_folder = r'D:\pycharm_projects\leaf\runs\detect\predict'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Loop through all files in the input folder
for file_name in os.listdir(input_folder):
    # Check if the file is an image (you can add more extensions if needed)
    if file_name.endswith(('.jpg', '.jpeg', '.png')):
        # Get the full path of the image
        image_path = os.path.join(input_folder, file_name)

        # Perform YOLO prediction
        results = model.predict(image_path, conf=0.25)

        # Get the image's save path
        save_path = os.path.join(output_folder, file_name)

        # Loop through results and save each image with bounding boxes
        for result in results:
            result.save(save_path)
