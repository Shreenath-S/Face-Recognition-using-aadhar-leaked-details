import cv2
import face_recognition
import pytesseract
from PIL import Image
import os

AADHAAR_DIR = 'C:\\Users\\Srinath\\OneDrive\\Others\\Desktop\\cn-19\\Data Sample\\Aadhar'
PAN_DIR = 'C:\\Users\\Srinath\\OneDrive\\Others\\Desktop\\cn-19\\Data Sample\\Pan'



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  


def recognize_face_and_extract_data(live_image_path):
    live_img = cv2.imread(live_image_path)
    rgb_live_img = cv2.cvtColor(live_img, cv2.COLOR_BGR2RGB)

    
    face_locations = face_recognition.face_locations(rgb_live_img)
    face_encodings = face_recognition.face_encodings(rgb_live_img, face_locations)

    match_found = False

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        found, card_type, card_path = compare_with_card_images(face_encoding)
        
        if found:
            match_found = True
            print(f"Match found with {card_type} card at {card_path}")
            extract_data_from_image(card_path)

        cv2.rectangle(live_img, (left, top), (right, bottom), (255, 0, 0), 2)
    
    if not match_found:
        print("No match found.")

    cv2.imshow('Detected Faces', live_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compare_with_card_images(face_encoding):
    for card_dir, card_type in [(AADHAAR_DIR, 'Aadhaar'), (PAN_DIR, 'PAN')]:
        for filename in os.listdir(card_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                card_image_path = os.path.join(card_dir, filename)
                card_img = face_recognition.load_image_file(card_image_path)
                card_face_encodings = face_recognition.face_encodings(card_img)
                
                for card_face_encoding in card_face_encodings:
                    match = face_recognition.compare_faces([card_face_encoding], face_encoding, tolerance=0.5)
                    if match[0]:
                        return True, card_type, card_image_path
    return False, None, None

def extract_data_from_image(image_path):
    img = preprocess_image_for_ocr(image_path)
    card_text = pytesseract.image_to_string(Image.fromarray(img), lang='eng')
    print(f"Extracted data from {image_path}:")
    print(card_text.strip())

    cv2.imshow(f"Extracted {image_path}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def preprocess_image_for_ocr(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return img

def capture_live_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("Press 'q' to capture the image.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        cv2.imshow("Webcam - Press 'q' to capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            captured_face_path = "captured_face.jpg"
            cv2.imwrite(captured_face_path, frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_face_path

if __name__ == "__main__":
    print("Capturing live face from webcam...")
    captured_face_image_path = capture_live_face()

    if captured_face_image_path:
        print("Recognizing captured face and extracting data...")
        recognize_face_and_extract_data(captured_face_image_path)
