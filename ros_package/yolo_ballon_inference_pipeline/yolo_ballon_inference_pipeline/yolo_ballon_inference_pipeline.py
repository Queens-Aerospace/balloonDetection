#!/usr/bin/env python3

import argparse
import cv2
import rospy
from sensor_msgs.msg import Image
from custom_msgs.msg import BoundingBoxes, BoundingBox  # Assuming a custom ROS message for bounding boxes
from ultralytics import YOLO
from cv_bridge import CvBridge
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Global variables
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for detections
bridge = CvBridge()

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Inference with ROS integration")
    parser.add_argument("--preview", "-p", action="store_true", help="Enable video preview with detections")
    return parser.parse_args()

def setup_ros():
    """Initialize ROS publishers and node."""
    rospy.init_node('yolo_inference_node', anonymous=True)
    image_pub = rospy.Publisher('/inferenced_image', Image, queue_size=10)
    bbox_pub = rospy.Publisher('/bounding_boxes', BoundingBoxes, queue_size=10)
    return image_pub, bbox_pub

def filter_detections(detections):
    """
    Filter detections based on confidence threshold.
    
    Args:
        detections (list): List of YOLO detection results.

    Returns:
        list: Filtered detections with bounds, confidences, and labels.
    """
    filtered = []
    for detection in detections:
        confidence = detection.conf.item() if hasattr(detection.conf, 'item') else detection.conf
        if confidence >= CONFIDENCE_THRESHOLD:
            filtered.append({
                'bounds': detection.boxes.xyxy.cpu().tolist()[0],
                'confidence': confidence,
                'label': detection.label if hasattr(detection, 'label') else None
            })
    return filtered

def publish_bounding_boxes(bbox_pub, detections):
    """
    Publish bounding box detections as a ROS message.
    
    Args:
        bbox_pub (Publisher): ROS publisher for bounding boxes.
        detections (list): List of filtered detection results.
    """
    bbox_msg = BoundingBoxes()
    bbox_msg.header.stamp = rospy.Time.now()
    for det in detections:
        box = BoundingBox()
        box.xmin, box.ymin, box.xmax, box.ymax = det['bounds']
        box.confidence = det['confidence']
        box.label = det['label']
        bbox_msg.boxes.append(box)
    bbox_pub.publish(bbox_msg)
    logging.info("Published bounding boxes with %d detections.", len(detections))

def main():
    args = parse_args()
    image_pub, bbox_pub = setup_ros()

    # Load YOLO model
    logging.info("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    logging.info("YOLO model loaded successfully.")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Error: Could not open the webcam.")
        return

    logging.info("Starting video feed. Press Ctrl+C to exit.")

    try:
        while not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read a frame from the webcam.")
                continue

            # Run YOLO inference
            results = model(frame)
            detections = filter_detections(results[0].boxes)

            # Publish bounding box detections
            publish_bounding_boxes(bbox_pub, detections)

            # Publish or preview image with detections
            if args.preview:
                annotated_frame = results[0].plot()  # Annotate frame with bounding boxes
                image_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
                image_pub.publish(image_msg)
                logging.info("Published annotated image.")

                # Display the frame for preview
                cv2.imshow("YOLO Inference Preview", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Preview stopped by user.")
                    break

    except rospy.ROSInterruptException:
        logging.info("ROS node interrupted.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logging.info("Webcam and windows released and closed.")

if __name__ == "__main__":
    main()
