#Advanced Face Recognition Program

This Python application utilizes face recognition technology to identify known and unknown faces in real-time using a webcam. It incorporates a graphical user interface (GUI) built with Tkinter and features such as email notifications, text-to-speech announcements, and statistical plotting of recognition counts.

#Features
Real-Time Face Recognition: Captures video from a webcam and identifies faces.
Database Integration: Stores and retrieves known face encodings from a SQLite database.
Email Notifications: Sends an email when a face is recognized.
Text-to-Speech: Announces the recognized name using text-to-speech.
Matched Face Display: Shows the matched face in a separate window.
Statistics Plotting: Plots and saves the recognition statistics as a graph.
User Management: Allows adding new known faces and viewing the list of known users.
Graphical User Interface: Provides an interactive GUI with Tkinter.

#Installation
Ensure you have the required libraries installed:
bash
pip install opencv-python face-recognition Pillow matplotlib smtplib pyttsx3 plyer


#Usage
Run the script to start the application:
bash
python advanced_face_recognition.py
