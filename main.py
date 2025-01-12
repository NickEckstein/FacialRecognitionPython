import threading

import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0) # Define Video Capture

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # Define Display Width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480) # Define Display Height

counter = 0

face_match = False

reference_img = cv2.imread("reference.jpg") # Load the Reference Image

def check_face(frame): # Function to check if the reference and the current fram match
    global face_match
    try:
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True
        else:
            face_match = False
    except ValueError:
        face_match = False
    

while True: # Pass a copy of the frame to the check_face function and check if there is a match or no match
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                threading.Thread(target=check_face, args=(frame.copy(),))
            except ValueError:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "NO MATCH", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key ==  ord(" "):
        break

cv2.destroyAllWindows()