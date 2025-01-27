import streamlit as st
import cv2
from ultralytics import YOLO

def main():
    st.title("YOLOv11 Real-time Object Detection")

    # Initialize YOLO model
    @st.cache_resource
    def load_model():
        return YOLO('best.pt')  # Load YOLOv11 model

    model = load_model()

    # Create a placeholder for webcam feed
    stframe = st.empty()

    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Add a stop button
    stop_button = st.button('Stop')

    while not stop_button:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam")
            break

        # Convert frame from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform detection
        results = model(frame_rgb)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Draw bounding box
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Add label
                label = f'{model.names[cls]} {conf:.2f}'
                cv2.putText(frame_rgb, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame
        stframe.image(frame_rgb, channels='RGB')

    # Release resources when stopped
    cap.release()


if __name__ == '__main__':
    main()