import cv2
from ultralytics import YOLO
import torch
from gtts import gTTS
import os
import pygame
import serial
import time
from threading import Thread
from queue import Queue
import numpy as np

class AudioManager:
    def __init__(self):
        pygame.mixer.init()
        self.audio_queue = Queue()
        self.audio_thread = Thread(target=self._audio_worker, daemon=True)
        self.audio_thread.start()
        self.audio_cache = {}  # Cache for audio files

    def _audio_worker(self):
        while True:
            text = self.audio_queue.get()
            filename = f"alert_{hash(text)}.mp3"
            
            # Generate new audio file
            tts = gTTS(text=text, lang='en')
            tts.save(filename)
            
            # Play the audio
            pygame.mixer.music.load(filename)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue
                
            # Delete file after playing
            try:
                os.remove(filename)
            except:
                pass

    def announce(self, text):
        self.audio_queue.put(text)

class PersonDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8m.pt').to(self.device)
        self.model2 = YOLO('weapons.pt').to(self.device)
        self.desired_classes1 = [0]
        self.desired_classes2 = [1]
        self.previous_count = 0
        self.audio_manager = AudioManager()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Arduino setup
        self.arduino = serial.Serial(port='COM3', baudrate=9600, timeout=1)
        self.last_sent_time = 0
        self.update_interval = 0.1
        self.count_start_time = time.time()
        self.temp_count = 0

    def send_to_arduino(self, xd, yd):
        try:
            data = f"{xd},{yd}\n"
            self.arduino.write(data.encode())
            time.sleep(0.1)
        except Exception as e:
            print(f"Error: {e}")

    def process_frame(self, frame):
        results1 = self.model.track(frame, persist=True, classes=self.desired_classes1, conf=0.6)
        results2 = self.model2.track(frame, persist=True, classes=self.desired_classes2, conf=0.2)
        current_count = len(results1[0].boxes)

        # Process detections
        aim_percentage_x = 0
        aim_percentage_y = 0
        for box in results1[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x = int((x1 + x2) / 2)
            y = int((y1 + y2) / 2)
            aim_percentage_x = (1 - x / 1280) * 100
            aim_percentage_y = (1 - y / 720) * 100

        # Handle person count changes with persistence check
        current_time = time.time()
        
        if current_count != self.temp_count:
            self.count_start_time = current_time
            self.temp_count = current_count
        elif current_time - self.count_start_time >= 1.0 and current_count != self.previous_count:
            if current_count > 0:
                text = f"{current_count} person{'s' if current_count > 1 else ''} detected."
                self.audio_manager.announce(text)
            else:
                self.audio_manager.announce("No persons detected.")
            self.previous_count = current_count

        combined_frame = frame.copy()
        annotated_frame1 = results1[0].plot()
        annotated_frame2 = results2[0].plot()
        
        # Combine the frames
        combined_frame = cv2.addWeighted(annotated_frame1, 0.5, annotated_frame2, 0.5, 0)


        return results1, results2, aim_percentage_x, aim_percentage_y, combined_frame

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            results1, results2, aim_percentage_x, aim_percentage_y, combined_frame = self.process_frame(frame)
            
            # Calculate aim degrees
            aim_degree_x = int((aim_percentage_x * 90) / 100)
            aim_degree_y = int((aim_percentage_y * 20) / 100)

            #Send to Arduino at intervals
            current_time = time.time()

            if current_time - self.last_sent_time >= self.update_interval:
                self.send_to_arduino(aim_degree_x, aim_degree_y)
                self.last_sent_time = current_time

            # Display frame
            cv2.imshow('YOLOv8 Webcam Detection', combined_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = PersonDetector()
    detector.run()