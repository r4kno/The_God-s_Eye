import cv2
from ultralytics import YOLO
import torch
from gtts import gTTS
import os
import pygame
import serial
import time

# Standard
pygame.mixer.init()

arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)

desired_classes = [0] 
previous_count = 0 
aim_percentage_x = 0
aim_percentage_y = 0

def announce(text):
    tts = gTTS(text=text, lang='en')
    tts.save("alert.mp3")

    if not pygame.mixer.get_init():
       pygame.mixer.init()
    

    pygame.mixer.music.load("alert.mp3")
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        continue
    
    # Stop music and release the file
    pygame.mixer.music.stop()
    pygame.mixer.quit()

    os.remove("alert.mp3")
def send_to_arduino(xd, yd):
    try:
        data = f"{xd},{yd}\n"  # Format data as comma-separated values
        arduino.write(data.encode())  # Send the data
        print(f"Sent: {data.strip()}")  # Log the sent data
        time.sleep(0.1)  # Allow Arduino to process
    except Exception as e:
        print(f"Error: {e}")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

model = YOLO('yolov8m.pt').to(device)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
last_sent_time = 0
update_interval = 0.1
    

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model.track(frame, persist=True, classes=desired_classes)
    person_count = len(results[0].boxes)  

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0]
        x = int((x1 + x2) / 2)
        y = int((y1 + y2) / 2)
        print(f"Person detected at ({x}, {y})")
        aim_percentage_x = (1- x / 1280) * 100
        aim_percentage_y = (1- y/720) * 100
   
    if person_count != previous_count:
        if person_count > 0:
            text = f"{person_count} person{'s' if person_count > 1 else ''} detected."
            announce(text)
        else:
            announce("No persons detected.")
    
    
    previous_count = person_count 
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)

    aim_percentage_z = 0
    aim_degree_x = int((aim_percentage_x * 90) / 100)
    aim_degree_y = int((aim_percentage_y * 20) / 100)
    aim_degree_z = (aim_percentage_z *100) / 100
    print(f"Aim Degree X: {aim_degree_x}")
    print(f"Aim Degree Y: {aim_degree_y}")


    #data sharing to arduino in an interval
    current_time = time.time()
    if current_time - last_sent_time >= update_interval:
        send_to_arduino(xd=aim_degree_x, yd=aim_degree_y)
        last_sent_time = current_time

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()