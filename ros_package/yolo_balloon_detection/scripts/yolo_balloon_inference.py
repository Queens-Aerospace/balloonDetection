#!/usr/bin/env python3

import argparse
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from yolo_balloon_detection.msg import BoundingBoxes, BoundingBox
from ultralytics import YOLO
from cv_bridge import CvBridge
import logging
import os

# Logging configuration
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Global variables
CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for detections
bridge = CvBridge()

CLASS_NAMES = {
        0: "red_ballon",
        1: "other",
        2: "other",
        3: "other",
    }


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLO Inference with ROS integration")
    parser.add_argument("--preview", "-p", action="store_true", help="Enable video preview with detections")
    parser.add_argument("--gui", "-g", action="store_true", help="Enable video gui preview with detections")
    return parser.parse_args()


class YOLOInferenceNode(Node):
    def __init__(self, enable_preview, enable_gui):
        super().__init__('yolo_inference_node')
        self.enable_preview = enable_preview
        self.enable_gui = enable_gui

        # Publishers
        self.image_pub = self.create_publisher(Image, '/inferenced_image', 10)
        self.bbox_pub = self.create_publisher(BoundingBoxes, '/bounding_boxes', 10)

        # YOLO Model
        self.get_logger().info("Loading YOLO model...")
        #self.model = YOLO("yolov8n.pt")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        yolo_path = os.path.join(script_dir, 'v00.pt')
        self.model = YOLO(yolo_path)
        self.get_logger().info("YOLO model loaded successfully.")

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open the webcam.")
            raise RuntimeError("Webcam not available")

        self.get_logger().info("Starting video feed. Press Ctrl+C to exit.")

        # Timer for periodic processing
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        """Process a single frame from the webcam."""
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to read a frame from the webcam.")
            return

        # Run YOLO inference
        results = self.model(frame)
        detections = self.filter_detections(results[0].boxes)

        # Publish bounding box detections
        self.publish_bounding_boxes(detections)

        # Publish or preview image with detections
        if self.enable_preview:
            annotated_frame = results[0].plot()  # Annotate frame with bounding boxes
            image_msg = bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")
            self.image_pub.publish(image_msg)
            self.get_logger().info("Published annotated image.")

        if self.enable_gui:
            cv2.imshow("YOLO Inference GUI", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.get_logger().info("Preview stopped by user.")
                rclpy.shutdown()

    def filter_detections(self, detections):
        """
        Filter detections based on confidence threshold.

        Args:
            detections (list): List of YOLO detection results.

        Returns:
            list: Filtered detections with bounds, confidences, and labels.
        """
        filtered = []
        for box in detections.data:  # Iterate through the detected boxes
            xmin, ymin, xmax, ymax, confidence, cls = box.tolist()
            if confidence >= CONFIDENCE_THRESHOLD:
                filtered.append({
                    'bounds': [xmin, ymin, xmax, ymax],
                    'confidence': confidence,
                    'label': int(cls),  # Convert class index to integer
                })
        return filtered

    def publish_bounding_boxes(self, detections):
        """
        Publish bounding box detections as a ROS message.

        Args:
            detections (list): List of filtered detection results.
        """

        bbox_msg = BoundingBoxes()
        bbox_msg.header.stamp = self.get_clock().now().to_msg()
        for det in detections:
            box = BoundingBox()
            box.xmin, box.ymin, box.xmax, box.ymax = det['bounds']
            box.confidence = det['confidence']
            box.label = CLASS_NAMES.get(det['label'], "unknown")  # Map ID to label or use "unknown"
            bbox_msg.boxes.append(box)
        self.bbox_pub.publish(bbox_msg)
        self.get_logger().info(f"Published bounding boxes with {len(detections)} detections.")


    def destroy_node(self):
        """Clean up resources."""
        super().destroy_node()
        self.cap.release()
        cv2.destroyAllWindows()
        self.get_logger().info("Webcam and windows released and closed.")


def main(args=None):
    # Parse command-line arguments
    cli_args = parse_args()

    # Initialize ROS
    rclpy.init(args=None)  # Pass None or sys.argv for ROS-specific args

    # Create the YOLO Inference Node
    try:
        yolo_node = YOLOInferenceNode(enable_preview=cli_args.preview, enable_gui=cli_args.gui)
        rclpy.spin(yolo_node)
    except RuntimeError as e:
        print(f"Node failed to initialize: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        yolo_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
