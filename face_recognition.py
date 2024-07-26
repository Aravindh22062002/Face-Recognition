import time
import cv2
import face_recognition
import os
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from PIL import Image, ImageTk
import sqlite3
import matplotlib.pyplot as plt
import smtplib
import pyttsx3
from email.message import EmailMessage
from plyer import notification

class AdvancedFaceRecognitionApp:
    def __init__(self):
        self.path = 'images'
        self.images = []
        self.classNames = []
        self.myList = os.listdir(self.path)
        
        for cl in self.myList:
            curImg = cv2.imread(f'{self.path}/{cl}')
            self.images.append(curImg)
            self.classNames.append(os.path.splitext(cl)[0])
        
        self.encodeListKnown = self.findEncodings(self.images)
        self.recognition_stats = {"known": 0, "unknown": 0}
        self.known_stats = []
        self.unknown_stats = []
        self.frame_count = 0

        # Initialize the Tkinter window
        self.root = tk.Tk()
        self.root.title("Advanced Face Recognition Program")

        # Create a label to display recognized name
        self.name_label = tk.Label(self.root, text="Recognized: ")
        self.name_label.pack()

        # Create a label to display a video feed
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        # Initialize the camera
        self.cap = cv2.VideoCapture(0)

        # Create buttons and labels for user interaction
        self.exit_button = tk.Button(self.root, text="Exit", command=self.exit_program)
        self.exit_button.pack()
        
        # Create labels to display recognition statistics
        self.stats_label = tk.Label(self.root, text="Recognition Statistics:")
        self.stats_label.pack()
        
        self.stats_label_known = tk.Label(self.root, text="Known Faces Recognized: 0")
        self.stats_label_known.pack()
        
        self.stats_label_unknown = tk.Label(self.root, text="Unknown Faces Recognized: 0")
        self.stats_label_unknown.pack()

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()
        
        # Initialize the variable to hold the last matched face image path
        self.last_matched_face_path = 'matchface'

        # Start recognizing and displaying the camera feed
        self.recognize_and_display()
        self.update_stats_labels()

        self.root.mainloop()

    def findEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList

    def open_file_dialog(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            image = cv2.imread(file_path)
            self.recognize_face(image)

    def add_known_face(self):
        name = simpledialog.askstring("Add Known Face", "Enter the name for the known face:")
        if name:
            file_path = filedialog.askopenfilename()
            if file_path:
                image = cv2.imread(file_path)
                encoding = face_recognition.face_encodings(image)
                if encoding:
                    self.encodeListKnown.append(encoding[0])
                    self.classNames.append(name)
                    # Save the encoding to the database
                    encoding_blob = encoding[0].tobytes()
                    self.cursor.execute('INSERT INTO users (name, encoding) VALUES (?, ?)', (name, encoding_blob))
                    self.conn.commit()
                    messagebox.showinfo("Success", "Known face added successfully!")

    def send_email(self, recognized_name):
        try:
            sender_email = "sindmahe@gmail.com"
            receiver_email = "mahe99522@gmail.com"
            password = "qjss ltxx hxid ptit"
            message = EmailMessage()
            message["From"] = sender_email
            message["To"] = receiver_email
            message["Subject"] = "Face Recognized"
            message.set_content(f"Face recognized: {recognized_name}")

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, password)
                server.send_message(message)
            print("Email sent successfully!")
        except smtplib.SMTPAuthenticationError as e:
            print(f"SMTP Authentication Error: {e}")
        except smtplib.SMTPException as e:
            print(f"SMTP Exception: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")

    def recognize_face(self, frame):
        imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(self.encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(self.encodeListKnown, encodeFace)
            matchindex = faceDis.argmin()

            if faceDis[matchindex] < 0.50:
                name = self.classNames[matchindex].upper()
                self.recognition_stats["known"] += 1

                # Send notification
                self.send_notification(name)
                # Display the matched face
                self.display_matched_face(frame, faceLoc)
                # Announce the recognized name
                self.speak_name(name)
                # Display the matched face for 5 seconds
                time.sleep(5)
            else:
                name = 'unknown'
                self.recognition_stats["unknown"] += 1
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                self.name_label.config(text="Recognized: " + name)

    def extract_matched_face(self, frame, face_location):
        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        matched_face = frame[y1:y2, x1:x2]
        matched_face_rgb = cv2.cvtColor(matched_face, cv2.COLOR_BGR2RGB)
        matched_face_path = os.path.join('matched_faces', 'last_matched_face.jpg')
        cv2.imwrite(matched_face_path, cv2.cvtColor(matched_face, cv2.COLOR_BGR2RGB))
        self.last_matched_face_path = matched_face_path
        return matched_face_rgb

    def view_users(self):
        user_list = self.cursor.execute('SELECT name FROM users').fetchall()
        users = [row[0] for row in user_list]
        users_str = '\n'.join(users)
        messagebox.showinfo("Known Users", f"Known users:\n{users_str}")

    def update_stats_labels(self):
        self.stats_label_known.config(text=f"Known Faces Recognized: {self.recognition_stats['known']}")
        self.stats_label_unknown.config(text=f"Unknown Faces Recognized: {self.recognition_stats['unknown']}")
        self.root.after(1000, self.update_stats_labels)

    def recognize_and_display(self):
        ret, frame = self.cap.read()
        if ret:
            self.frame_count += 1
            self.recognize_face(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            self.video_label.config(image=img)
            self.video_label.image = img
            self.root.after(10, self.recognize_and_display)
            if self.frame_count % 100 == 0:
                self.update_plot()

    def send_notification(self, name):
        notification_title = "Face Recognized"
        notification_message = f"Face recognized: {name}"
        notification.notify(
            title=notification_title,
            message=notification_message,
            app_name="Advanced Face Recognition",
        )

    def display_matched_face(self, frame, face_location):
        y1, x2, y2, x1 = face_location
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        matched_face = frame[y1:y2, x1:x2]
        cv2.imshow('Matched Face', matched_face)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    def speak_name(self, name):
        self.engine.say(f"Face matched. The recognized name is {name}")
        self.engine.runAndWait()

    def update_plot(self):
        self.known_stats.append(self.recognition_stats["known"])
        self.unknown_stats.append(self.recognition_stats["unknown"])
        plt.figure(figsize=(8, 6))
        plt.plot(self.known_stats, label='Known Faces Recognized', marker='o')
        plt.plot(self.unknown_stats, label='Unknown Faces Recognized', marker='o')
        plt.xlabel('Frame Count (x100)')
        plt.ylabel('Recognition Count')
        plt.legend()
        plt.title('Face Recognition Statistics')
        plt.grid()
        plt.tight_layout()
        plt.savefig('recognition_stats.png')
        plt.close()

    def exit_program(self):
        self.cap.release()
        self.conn.close()
        self.root.quit()

if __name__ == "__main__":
    app = AdvancedFaceRecognitionApp()
