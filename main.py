# libraries
import cv2
import csv
from datetime import datetime
from simple_face_rec import SimpleFacerec

# Encoding Images
sfr = SimpleFacerec()
known_names = sfr.load_encoding_images("students/").copy()

# CSV file
now = datetime.now()
date = now.strftime("%d-%m-%Y")
file = open("attendance/"+date+".csv",'w+',newline='')
writer = csv.writer(file)
writer.writerow(['Name', 'Time'])

# Load Camera
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # Detect Faces
    face_locations, names = sfr.detect_known_faces(frame)
    # currentStudents = names.copy()
    for face_loc, name in zip(face_locations, names):
        y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(frame, name, (x2, y1-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,0), 1)

        # Add names in CSV file
        if name in known_names:
            known_names.remove(name)
            time = now.strftime("%H:%M:%S")
            writer.writerow([name, time])

    cv2.imshow("Camera",frame)
    if cv2.waitKey(1) == 27: # for 'esc' key
        break
cap.release()
cv2.destroyAllWindows()
file.close()