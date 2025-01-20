import cv2
from ultralytics import YOLO

def main():
    # Load the YOLO model
    model = YOLO("yolov8n.pt")  # You can replace 'yolov8n.pt' with the model you want (e.g., yolov8m.pt, yolov8l.pt)

    # Open the webcam feed
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera; replace with the camera index or video path if needed

    if not cap.isOpened():
        print("Error: Could not open the webcam.")
        return

    print("Press 'q' to exit.")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read a frame from the webcam.")
            break

        # Run YOLO inference on the frame
        results = model(frame)

        # Visualize detections on the frame
        annotated_frame = results[0].plot()  # The 'plot()' method draws the detections on the frame

        # Display the frame with detections
        cv2.imshow("YOLO Object Detection", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
