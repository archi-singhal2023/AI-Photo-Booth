# AI Photo Booth 📸✨

An interactive AI-powered photo booth built with Python that captures photos using:
- 👀 Blink detection (via MediaPipe Face Mesh)
- ✋ Finger detection (via MediaPipe Hands)

Captured images are compiled into a **polaroid-style photo strip**.

## Features
- Real-time blink/finger trigger to snap photos
- Countdown with sound effects
- Final polaroid strip generation
- Built with OpenCV, MediaPipe, and Tkinter

## 🧠 Features

- 👀 **Blink Detection**: Automatically captures photos when the user blinks.
- ☝️ **Finger Detection**: Filter changes when the finger is detected upon filter button 
- 🎞️ **Photo Strip Generator**: Compiles 3 photos into a clean polaroid-style strip.
- 🔇 **Voice Feedback**: Audible countdown with sound cues (1, 2, 3... SNAP!)
- 🖼️ **Stylized Output**: Resizes images with a polaroid frame and displays the final strip.
- 🧘‍♀️ **User-friendly**: No manual button clicking; just use your face and hand gestures.

## How to Run
```bash
python ai_booth.py
