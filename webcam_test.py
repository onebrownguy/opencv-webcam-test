import cv2

# Open the webcam (try changing 0 to 1 or 2 if needed)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Webcam Test', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
