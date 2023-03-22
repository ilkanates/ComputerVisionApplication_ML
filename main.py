import csv
import datetime
import cv2
from face_recog import FaceRecognition


# Encode faces from a folder
sfr = FaceRecognition()
sfr.load_encoding_images("images/")

# Open CSV file to append data
with open('names.csv', mode='a', newline='') as names_file:
    fieldnames = ['Name', 'Time']
    csv_writer = csv.DictWriter(names_file, fieldnames=fieldnames)

    # Write header if file is empty
    if names_file.tell() == 0:
        csv_writer.writeheader()

    # Load Camera
    cap = cv2.VideoCapture(0)

    # Set the time limit to 5 minutes
    time_limit = datetime.timedelta(minutes=1)

    # Initialize the last seen times for each person
    last_seen_times = {}

    while True:
        ret, frame = cap.read()
        names = []

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)
        for face_loc, name in zip(face_locations, face_names):
            y1, x1, y2, x2 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
            names.append(name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 1)

        cv2.imshow("Frame", frame)

        # Write names and timestamps to CSV file
        current_time = datetime.datetime.now()
        for name in names:
            # Check if the person was seen within the last 5 minutes
            if name in last_seen_times and current_time - last_seen_times[name] <= time_limit:
                continue

            # Write the name and current timestamp to the CSV file
            csv_writer.writerow({'Name': name, 'Time': current_time})
            last_seen_times[name] = current_time
        print(names, current_time)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()