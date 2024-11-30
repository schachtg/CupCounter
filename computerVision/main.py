import cv2
import torch
from ultralytics import YOLO
import numpy as np
import random
from datetime import datetime
import os
import shutil
import threading

class VideoStream:
    def __init__(self, source):
        self.stream = cv2.VideoCapture(source)
        self.grabbed, self.frame = self.stream.read()
        self.started = False
        self.read_lock = threading.Lock()
        
    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self
    
    def update(self):
        while self.started:
            grabbed, frame = self.stream.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                
    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
        return self.grabbed, frame
    
    def release(self):
        self.started = False
        self.thread.join()
        self.stream.release()

    def isOpened(self):
        return self.stream.isOpened()

# Define the output directory
output_dir = ""

# Clear the output directory
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

model = YOLO("weights/yolov8x.pt", "v8")

# Generate unique colors for each label
labels = model.names
color_map = {label: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for label in labels}

videos = []
closeWindows = False
cupHoldingThreshold = 2.6
savedFramesCount = 0

# Open each video
#videos.append(cv2.VideoCapture("FilePath"))
#videos.append(cv2.VideoCapture(0))

def trigger_alarm(alert_type, bbox, image):
    global savedFramesCount
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255)
    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    # Get the current date and time
    now = datetime.now()

    # Format the date and time
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    savedFramesCount += 1

    frame_name = 'Frame'+str(savedFramesCount)+'.jpg'
    out_path = f"{output_dir}/{frame_name}"
    cv2.imwrite(out_path, image)
    print(formatted_now, "Frame saved to ", out_path)


    if alert_type == "weapon":
        print("Weapon detected!")
    elif alert_type == "cups":
        print("Person holding too many cups detected!")

def nms_pytorch(P : torch.tensor ,thresh_iou : float):
    """
    Apply non-maximum suppression to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the image 
            along with the class predscores, Shape: [num_boxes,5].
        thresh_iou: (float) The overlap thresh for suppressing unnecessary boxes.
    Returns:
        A list of filtered boxes, Shape: [ , 5]
    """
 
    # we extract coordinates for every 
    # prediction box present in P
    x1 = P[:, 0]
    y1 = P[:, 1]
    x2 = P[:, 2]
    y2 = P[:, 3]
 
    # we extract the confidence scores as well
    scores = P[:, 4]
 
    # calculate area of every block in P
    areas = (x2 - x1) * (y2 - y1)
     
    # sort the prediction boxes in P
    # according to their confidence scores
    order = scores.argsort()
 
    # initialise an empty list for 
    # filtered prediction boxes
    keep = []
     
 
    while len(order) > 0:
         
        # extract the index of the 
        # prediction with highest score
        # we call this prediction S
        idx = order[-1]
 
        # push S in filtered predictions list
        keep.append(P[idx])
 
        # remove S from P
        order = order[:-1]
 
        # sanity check
        if len(order) == 0:
            break
         
        # select coordinates of BBoxes according to 
        # the indices in order
        xx1 = torch.index_select(x1,dim = 0, index = order)
        xx2 = torch.index_select(x2,dim = 0, index = order)
        yy1 = torch.index_select(y1,dim = 0, index = order)
        yy2 = torch.index_select(y2,dim = 0, index = order)
 
        # find the coordinates of the intersection boxes
        xx1 = torch.max(xx1, x1[idx])
        yy1 = torch.max(yy1, y1[idx])
        xx2 = torch.min(xx2, x2[idx])
        yy2 = torch.min(yy2, y2[idx])
 
        # find height and width of the intersection boxes
        w = xx2 - xx1
        h = yy2 - yy1
         
        # take max with 0.0 to avoid negative w and h
        # due to non-overlapping boxes
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
 
        # find the intersection area
        inter = w*h
 
        # find the areas of BBoxes according the indices in order
        rem_areas = torch.index_select(areas, dim = 0, index = order) 
 
        # find the union of every prediction T in P
        # with the prediction S
        # Note that areas[idx] represents area of S
        union = (rem_areas - inter) + areas[idx]
         
        # find the IoU of every prediction in P with S
        IoU = inter / union
 
        # keep the boxes with IoU less than thresh_iou
        mask = IoU < thresh_iou
        order = order[mask]
     
    return keep

def do_boxes_overlap(rect1, rect2):
    x1A, y1A, x2A, y2A = rect1
    x1B, y1B, x2B, y2B = rect2

    # Check for horizontal overlap
    horizontal_overlap = (x1A < x2B) and (x2A > x1B)

    # Check for vertical overlap
    vertical_overlap = (y1A < y2B) and (y2A > y1B)

    # Rectangles overlap if both horizontal and vertical projections overlap
    return horizontal_overlap and vertical_overlap

for video in videos:
    if not video.isOpened():
        print("Could not open video")
        closeWindows = True

while not closeWindows:
    frames = []
    for video in videos:
        ret, frame = video.read()
        if not ret:
            closeWindows = True
            break

        frames.append(frame)

    if not closeWindows:
        for frame in frames:
            cups = []
            people = []
            frameCopy = frame.copy()
            # Perform object detection
            results = model(frame)
            boxes = []

            if len(results) > 0:
                # Get bounding boxes
                for result in results:
                    for box in result.boxes:
                        # Convert coordinates to integers
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf.item()
                        class_id = box.cls.item()
                        boxes.append([x1, y1, x2, y2, confidence, class_id])
                
                boxes_tensor = torch.tensor(boxes)

                if len(boxes_tensor) > 0:
                    # Apply non-maximum suppression
                    filtered_boxes = nms_pytorch(boxes_tensor, 0.3)

                    # Draw bounding boxes
                    for box in filtered_boxes:
                        x1, y1, x2, y2, confidence, class_id = box
                        label = model.names[int(class_id)]

                        # Get color for the label
                        color = color_map[int(class_id)]

                        if label == "cup" or label == "bottle":
                            cups.append([x1, y1, x2, y2])
                        elif label == "person":
                            people.append([x1, y1, x2, y2])
                        elif label == "knife" or label == "gun" or label == "scissors":
                            trigger_alarm("weapon", [x1, y1, x2, y2], frame)

                        # Draw rectangle and label on the frame
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
            # Get estimated number of cups held by each person
            num_potential_holders = 0
            potential_holders_indeces = []
            held_confidence = [0 for _ in people]
            for cup in cups:
                num_potential_holders = 0
                potential_holders_indeces = []
                for index, person in enumerate(people):
                    if do_boxes_overlap(person, cup):
                        num_potential_holders += 1
                        potential_holders_indeces.append(index)
                if num_potential_holders > 0:
                    cv2.rectangle(frameCopy, (int(cup[0]), int(cup[1])), (int(cup[2]), int(cup[3])), (255,255,255), 2)
                    for potential_holder_index in potential_holders_indeces:
                        held_confidence[potential_holder_index] += 1/num_potential_holders


            for index, person in enumerate(people):
                if held_confidence[index] >= cupHoldingThreshold:
                    highlighted_target = frameCopy.copy()
                    print("Estimated holding confidence: ", held_confidence[index])
                    trigger_alarm("cups", person, highlighted_target)  
                

    # Loop through the frames and display each in a separate window
    for i, frame in enumerate(frames):
        window_name = f"Camera {i+1}"
        cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the video capture object
video.release()

# Closes all the frames
cv2.destroyAllWindows()
