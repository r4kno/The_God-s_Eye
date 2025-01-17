import cv2
from ultralytics import YOLO
import torch

#Distance estimation

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = YOLO('yolov8m.pt').to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def estimate_distance(known_height, focal_length, perceived_height_pixels):
    if perceived_height_pixels == 0:
        return None
    distance = (known_height * focal_length) / perceived_height_pixels
    return distance


known_person_height = 1.7 
focal_length_in_pixels = 800 

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model.track(frame, persist=True, classes=[0]) 
    if results:
        for result in results[0].boxes:  
        
            bbox = result.xyxy[0]  
            perceived_height_of_person = bbox[3] - bbox[1]  #(ymax - ymin)
            
            # Estimate the distance
            distance = estimate_distance(known_person_height, focal_length_in_pixels, perceived_height_of_person)
            print(f"Estimated distance to person: {distance:.2f} meters")

           
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f'Distance: {distance:.2f}m', (int(bbox[0]), int(bbox[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
