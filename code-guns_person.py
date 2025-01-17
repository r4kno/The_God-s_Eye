import cv2
from ultralytics import YOLO
import torch
from gtts import gTTS
import os
import pygame

# Standard with guns
pygame.mixer.init()

def announce(text):
    tts = gTTS(text=text, lang='en')
    tts.save("alert.mp3")

    if not pygame.mixer.get_init():
       pygame.mixer.init()
    

    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        continue
    
    # Stop the music and release the file
    pygame.mixer.music.stop()
    pygame.mixer.quit()

    os.remove("alert.mp3")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model_person = YOLO('yolov8m.pt').to(device) #person model
model_gun = YOLO('yolov8n.pt').to(device) #gun model

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

desired_classes = [0] 
previous_count = 0 
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    
    # Detect guns
    results_gun = model_gun.track(frame, persist=True, classes=[0])
    gun_count = len(results_gun[0].boxes)

    # Detect persons
    results_person = model_person.track(frame, persist=True, classes=[0]) 
    person_count = len(results_person[0].boxes)  
    if gun_count > 0:
        announce(f"{gun_count} gun{'s' if gun_count > 1 else ''} detected.")
    
    if person_count > 0:
        announce(f"{person_count} person{'s' if person_count > 1 else ''} detected.")
    
    annotated_frame = frame.copy() 

    if gun_count > 0:
        annotated_frame = results_gun[0].plot()

    if person_count > 0:
        annotated_frame = results_person[0].plot(annotated_frame)

    cv2.imshow('YOLOv8 Multi-Model Detection', annotated_frame)


    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()