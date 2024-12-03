import cv2
import random
import datetime
import string
import numpy as np
import face_recognition


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def generate_random_tamil_name():
    first_names = ["John", "Jane", "Alex", "Emily", "Chris", "Katie", "Michael", "Sarah"]
    last_names = ["Doe", "Smith", "Johnson", "Brown", "Williams", "Jones", "Davis", "Garcia"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def generate_random_aadhaar():
    return ''.join([str(random.randint(0, 9)) for _ in range(12)])

def generate_random_pan():
    return ''.join(random.choices(string.ascii_uppercase, k=5)) + str(random.randint(1000, 9999)) + random.choice(string.ascii_uppercase)

face_data_store = {}
person_id_counter = 1  

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def recognize_faces(image, faces):
    global person_id_counter
    rgb_image = image[:, :, ::-1] 
    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_encoding_tuple = tuple(face_encoding)  
        matches = face_recognition.compare_faces(list(face_data_store.keys()), face_encoding, tolerance=0.6)
        name, aadhaar, pan, person_id = "Unknown", "", "", None

        if any(matches):
            first_match_index = matches.index(True)
            face_encoding_key = list(face_data_store.keys())[first_match_index]
            name, aadhaar, pan, person_id = face_data_store[face_encoding_key]
        else:
            name = generate_random_tamil_name()
            aadhaar = generate_random_aadhaar()
            pan = generate_random_pan()
            person_id = person_id_counter
            face_data_store[face_encoding_tuple] = (name, aadhaar, pan, person_id)
            person_id_counter += 1

        # Ensure only one person is associated with this face area
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"Person ID: {person_id}\nName: {name}\nAadhaar: {aadhaar}\nPAN: {pan}"
        lines = text.split('\n')
        y0, dy = top - 20, 20
        for i, line in enumerate(lines):
            y = y0 + i * dy
            cv2.putText(image, line, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Check for data leaks
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        face_encoding_tuple = tuple(face_encoding)
        if face_encoding_tuple in face_data_store:
            stored_name, stored_aadhaar, stored_pan, _ = face_data_store[face_encoding_tuple]
            current_name, current_aadhaar, current_pan, _ = name, aadhaar, pan, person_id
            if stored_name != current_name or stored_aadhaar != current_aadhaar or stored_pan != current_pan:
                cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)

    return image


def process_frame(frame):
    faces = detect_faces(frame)
    annotated_frame = recognize_faces(frame, faces)
    return annotated_frame

def run_recognition():
    video_capture = cv2.VideoCapture(0)  

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        annotated_frame = process_frame(frame)
        cv2.imshow('Video', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_recognition()
