import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading
import time
import os
import numpy as np
import pygame

# Init MediaPipe & pygame
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False)
hands = mp_hands.Hands(max_num_hands=1)
pygame.mixer.init()

# Audio
def play_audio(file_path):
    try:
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            continue
    except Exception as e:
        print(f"Audio error: {e}")

def countdown_audio():
    for word in ["three.wav", "two.wav", "one.wav"]:
        play_audio(word)
        time.sleep(0.5)

# Filters
filters = ["None", "Gray", "Sepia", "Invert", "Sketch"]
def apply_filter(image, name):
    if name == "Gray":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif name == "Sepia":
        sepia = cv2.transform(np.array(image, dtype=np.float64),
            np.matrix([[0.393, 0.769, 0.189],
                       [0.349, 0.686, 0.168],
                       [0.272, 0.534, 0.131]]))
        return np.clip(sepia, 0, 255).astype(np.uint8)
    elif name == "Invert":
        return cv2.bitwise_not(image)
    elif name == "Sketch":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv = 255 - gray
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch = cv2.divide(gray, 255 - blur, scale=256)
        return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
    return image

class PhotoBoothApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Photo Booth")
        self.video = cv2.VideoCapture(0)
        self.canvas = tk.Label(root)
        self.canvas.pack()

        # States
        self.filter_index = 0
        self.blinked = False
        self.blink_counter = 0
        self.blink_threshold = 0.21
        self.required_frames = 2
        self.face_detected_time = None
        self.countdown_active = False
        self.countdown_start = None
        self.blink_triggered = False

        self.photo_count = 0
        self.max_photos = 3
        self.photos = []
        self.photo_delay = 3
        self.show_strip = False
        self.save_path = "photos"
        os.makedirs(self.save_path, exist_ok=True)

        self.filter_button_coords = (10, 400, 150, 440)
        self.save_button_coords = (180, 400, 320, 440)
        self.last_tap_time = 0

        self.update_frame()

    def calculate_ear(self, eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def next_filter(self):
        self.filter_index = (self.filter_index + 1) % len(filters)

    def save_strip(self):
        if len(self.photos) < self.max_photos:
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        if path:
            strip = self.create_polaroid_strip(self.photos)
            cv2.imwrite(path, strip)
            print(f"Saved at {path}")

    def detect_finger_tap(self, x, y):
        now = time.time()
        if now - self.last_tap_time < 1: return
        fx1, fy1, fx2, fy2 = self.filter_button_coords
        sx1, sy1, sx2, sy2 = self.save_button_coords
        if fx1 <= x <= fx2 and fy1 <= y <= fy2:
            self.next_filter()
            self.last_tap_time = now
        elif sx1 <= x <= sx2 and sy1 <= y <= sy2 and len(self.photos) == self.max_photos:
            self.save_strip()
            self.last_tap_time = now

    def update_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_result = face_mesh.process(rgb)
        hand_result = hands.process(rgb)

        if face_result.multi_face_landmarks:
            if self.face_detected_time is None:
                self.face_detected_time = time.time()

            for landmarks in face_result.multi_face_landmarks:
                left_eye = [landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                right_eye = [landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                left_eye = np.array([(pt.x * w, pt.y * h) for pt in left_eye])
                right_eye = np.array([(pt.x * w, pt.y * h) for pt in right_eye])
                ear = (self.calculate_ear(left_eye) + self.calculate_ear(right_eye)) / 2.0

                if ear < self.blink_threshold and not self.show_strip and not self.countdown_active:
                    self.blink_counter += 1
                    if self.blink_counter >= self.required_frames:
                        print("Blink Detected 👁️")
                        self.countdown_active = True
                        self.countdown_start = time.time()
                        self.photos = []
                        threading.Thread(target=countdown_audio).start()
                else:
                    self.blink_counter = 0

        if self.countdown_active and not self.show_strip:
            elapsed = time.time() - self.countdown_start
            if elapsed >= self.photo_delay:
                play_audio("snap.wav")
                filename = f"photo_{len(self.photos)+1}.jpg"
                cv2.imwrite(filename, frame)
                self.photos.append(cv2.imread(filename))
                if len(self.photos) >= self.max_photos:
                    self.countdown_active = False
                    self.show_strip = True
                    strip = self.create_polaroid_strip(self.photos)
                    cv2.imshow("Your Photo Strip 🎞️", strip)
                    cv2.imwrite("final_strip.jpg", strip)
                    cv2.setWindowProperty("Your Photo Strip 🎞️", cv2.WND_PROP_TOPMOST, 1)
                    cv2.waitKey(1)
                else:
                    self.countdown_start = time.time()

        display = apply_filter(frame.copy(), filters[self.filter_index])

        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                index_tip = hand_landmarks.landmark[8]
                fx, fy = int(index_tip.x * w), int(index_tip.y * h)
                cv2.circle(display, (fx, fy), 12, (0, 0, 255), -1)
                self.detect_finger_tap(fx, fy)

        # Buttons
        cv2.rectangle(display, self.filter_button_coords[:2], self.filter_button_coords[2:], (0, 255, 0), -1)
        cv2.putText(display, "Filter", (self.filter_button_coords[0]+10, self.filter_button_coords[1]+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        color = (0, 255, 0) if len(self.photos) == self.max_photos else (100, 100, 100)
        cv2.rectangle(display, self.save_button_coords[:2], self.save_button_coords[2:], color, -1)
        cv2.putText(display, "Save", (self.save_button_coords[0]+10, self.save_button_coords[1]+28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        img = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)
        self.root.after(10, self.update_frame)

        # Handle closing of photo strip
        if self.show_strip and cv2.getWindowProperty("Your Photo Strip 🎞️", cv2.WND_PROP_VISIBLE) < 1:
            self.show_strip = False
            self.photos.clear()
            print("Strip closed. Ready for next session.")

    def create_polaroid_strip(self, images):
        polaroids = []
        for img in images:
            img = cv2.resize(img, (180, 180))
            frame = 255 * np.ones((230, 200, 3), dtype=np.uint8)  # white background
            frame[10:190, 10:190] = img  # leave bottom area blank for polaroid effect
            polaroids.append(frame)
        return cv2.vconcat(polaroids)


if __name__ == "__main__":
    root = tk.Tk()
    app = PhotoBoothApp(root)
    root.mainloop()
