import cv2

# Load the Haar cascades for face and eyeglasses detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
glasses_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(40, 40))

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            # Detect glasses in the face ROI
            glasses = glasses_cascade.detectMultiScale(roi_gray, scaleFactor=1.2, minNeighbors=10, minSize=(20, 20))

            if len(glasses) > 0:
                cv2.putText(frame, "Glasses Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                color = (0, 255, 0)  # Green for detected glasses
            else:
                cv2.putText(frame, "No Glasses", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                color = (0, 0, 255)  # Red for no glasses

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)

        # Show the video feed
        cv2.imshow("Glasses Detection", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
