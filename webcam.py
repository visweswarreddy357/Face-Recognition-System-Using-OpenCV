import numpy as np
import face_recognition as fr
import cv2
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
image = fr.load_image_file("Mahesh babu.jpg")
face_encoding = fr.face_encodings(image)[0]
known_face_encondings = [face_encoding]
known_face_names = ["Mahesh Babu"]
while True:
    ret, frame = cap.read()
    rgb_frame = frame[:, :, ::-1]
    face_locations = fr.face_locations(rgb_frame)
    face_encodings = fr.face_encodings(rgb_frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face_encondings, face_encoding)
        name = "Unknown"
        face_distances = fr.face_distance(known_face_encondings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom -35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Webcam_facerecognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()