import cv2
import numpy as np
import imutils
from collections import deque
import os
import urllib.request


class VesselMouldDetector:
    def __init__(self):
        # Define Mould color range in HSV
        self.lower_Mould = np.array([100, 50, 50])  # Blue-ish color for Mould
        self.upper_Mould = np.array([140, 255, 255])

        # For tracking Mould level points
        self.pts = deque(maxlen=64)

        # COCO class labels that MobileNet SSD was trained on
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "Liquid_Metal_Level", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # Target vessel classes we want to detect
        self.TARGET_VESSELS = ['Liquid_Metal_Level', 'cup', 'bowl', 'wine glass', 'vase']

        # Colors for drawing bounding boxes (BGR format)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        # Initialize the neural network
        self.net = None
        self.load_mobilenet_ssd()

    def load_mobilenet_ssd(self):
        """Load MobileNet SSD model for object detection"""
        try:
            # Try to load local model files
            prototxt_path = "MobileNetSSD_deploy.prototxt"
            model_path = "MobileNetSSD_deploy.caffemodel"

            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                print("[INFO] Downloading MobileNet SSD model files...")
                self.download_mobilenet_ssd()

            print("[INFO] Loading MobileNet SSD model...")
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("[INFO] Model loaded successfully!")

        except Exception as e:
            print(f"[ERROR] Failed to load MobileNet SSD model: {e}")
            print("[INFO] Falling back to basic contour detection...")
            self.net = None

    def download_mobilenet_ssd(self):
        """Download MobileNet SSD model files"""
        # URLs for MobileNet SSD model files
        prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        model_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"

        try:
            # Download prototxt file
            print("[INFO] Downloading prototxt file...")
            urllib.request.urlretrieve(prototxt_url, "MobileNetSSD_deploy.prototxt")

            # Note: The model file is large and requires manual download
            print("[WARNING] Please manually download the MobileNetSSD_deploy.caffemodel file")
            print("[WARNING] from: https://github.com/chuanqi305/MobileNet-SSD")

        except Exception as e:
            print(f"[ERROR] Failed to download model files: {e}")

    def detect_vessels(self, image):
        """Detect vessels (Liquid_Metal_Levels, cups, bowls, etc.) in the image"""
        if self.net is None:
            return []

        (h, w) = image.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                     (300, 300), 127.5)

        # Pass blob through network
        self.net.setInput(blob)
        detections = self.net.forward()

        vessels = []

        # Loop over detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter weak detections
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])

                # Check if detected object is a vessel type
                if idx < len(self.CLASSES):
                    class_name = self.CLASSES[idx]

                    # Check if it's a vessel we're interested in
                    if class_name in ["Liquid_Metal_Level", "cup", "bowl"] or "table" in class_name.lower():
                        # Extract bounding box
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        vessels.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': (startX, startY, endX, endY),
                            'center': ((startX + endX) // 2, (startY + endY) // 2)
                        })

        return vessels

    def detect_Liquid_Metal_Level(self, image):
        """Detect Mould Liquid_Metal_Level in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply threshold
        _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (presumably the Liquid_Metal_Level)
        max_area = 0
        Liquid_Metal_Level_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                Liquid_Metal_Level_contour = contour

        return Liquid_Metal_Level_contour, thresh

    def detect_Mould_level_in_vessel(self, image, vessel_bbox):
        """Detect Mould level within a specific vessel"""
        startX, startY, endX, endY = vessel_bbox

        # Extract vessel region
        vessel_roi = image[startY:endY, startX:endX]

        if vessel_roi.size == 0:
            return None, None

        # Convert to HSV and detect Mould
        hsv = cv2.cvtColor(vessel_roi, cv2.COLOR_BGR2HSV)

        # Expand Mould detection range for better detection
        lower_Mould = np.array([100, 30, 30])  # More permissive blue range
        upper_Mould = np.array([140, 255, 255])

        Mould_mask = cv2.inRange(hsv, lower_Mould, upper_Mould)

        # Also detect darker Mould/liquid
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

        # Combine masks
        combined_mask = cv2.bitwise_or(Mould_mask, dark_mask)

        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the Mould mask
        Mould_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(Mould_contours) > 0:
            # Find the largest Mould contour
            Mould_contour = max(Mould_contours, key=cv2.contourArea)

            if cv2.contourArea(Mould_contour) > 50:  # Minimum area threshold
                # Find the topmost point of the Mould contour
                topmost = tuple(Mould_contour[Mould_contour[:, :, 1].argmin()][0])

                # Convert back to original image coordinates
                Mould_level_x = startX + topmost[0]
                Mould_level_y = startY + topmost[1]

                # Calculate Mould level percentage
                vessel_height = endY - startY
                Mould_height = endY - Mould_level_y
                Mould_percentage = (Mould_height / vessel_height) * 100 if vessel_height > 0 else 0

                return {
                    'position': (Mould_level_x, Mould_level_y),
                    'percentage': Mould_percentage,
                    'vessel_height': vessel_height,
                    'Mould_height': Mould_height
                }, combined_mask

        return None, combined_mask

    def classify_vessel_type(self, vessel_class, bbox):
        """Classify vessel type and provide specific detection parameters"""
        startX, startY, endX, endY = bbox
        width = endX - startX
        height = endY - startY
        aspect_ratio = height / width if width > 0 else 0

        vessel_info = {
            'type': vessel_class,
            'shape': 'unknown',
            'expected_Mould_region': 'bottom'
        }

        if vessel_class == 'Liquid_Metal_Level':
            if aspect_ratio > 2.0:
                vessel_info['shape'] = 'tall_Liquid_Metal_Level'
            elif aspect_ratio > 1.5:
                vessel_info['shape'] = 'standard_Liquid_Metal_Level'
            else:
                vessel_info['shape'] = 'wide_Liquid_Metal_Level'
            vessel_info['expected_Mould_region'] = 'bottom_to_neck'

        elif vessel_class == 'cup':
            vessel_info['shape'] = 'cylindrical'
            vessel_info['expected_Mould_region'] = 'bottom_portion'

        elif 'bowl' in vessel_class:
            vessel_info['shape'] = 'wide_shallow'
            vessel_info['expected_Mould_region'] = 'center_bottom'

        elif 'table' in vessel_class:
            vessel_info['shape'] = 'surface'
            vessel_info['expected_Mould_region'] = 'on_surface'

        return vessel_info

    def process_frame(self, frame):
        """Process a video frame to detect vessels and Mould levels"""
        # Make a copy for drawing
        output = frame.copy()

        # Detect vessels using MobileNet SSD
        vessels = self.detect_vessels(frame)

        # If no vessels detected with neural network, fall back to contour detection
        if not vessels:
            Liquid_Metal_Level_contour, thresh = self.detect_Liquid_Metal_Level(frame)
            if Liquid_Metal_Level_contour is not None:
                # Get bounding box from contour
                x, y, w, h = cv2.boundingRect(Liquid_Metal_Level_contour)
                vessels = [{
                    'class': 'Liquid_Metal_Level',
                    'confidence': 0.8,
                    'bbox': (x, y, x + w, y + h),
                    'center': (x + w // 2, y + h // 2)
                }]
        else:
            thresh = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Process each detected vessel
        for vessel in vessels:
            vessel_class = vessel['class']
            confidence = vessel['confidence']
            bbox = vessel['bbox']
            startX, startY, endX, endY = bbox

            # Classify vessel type for better Mould detection
            vessel_info = self.classify_vessel_type(vessel_class, bbox)

            # Draw vessel bounding box
            color = [int(c) for c in self.COLORS[self.CLASSES.index(vessel_class)]]
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

            # Add vessel label
            label = f"{vessel_class}: {confidence:.2f}"
            cv2.putText(output, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Detect Mould level in this vessel
            Mould_info, Mould_mask = self.detect_Mould_level_in_vessel(frame, bbox)

            if Mould_info is not None:
                Mould_pos = Mould_info['position']
                Mould_percentage = Mould_info['percentage']

                # Draw Mould level point
                cv2.circle(output, Mould_pos, 5, (255, 0, 0), -1)

                # Add Mould level information
                Mould_text = f"Mould: {Mould_percentage:.1f}%"
                cv2.putText(output, Mould_text,
                            (startX, endY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Draw Mould level line across vessel
                cv2.line(output, (startX, Mould_pos[1]), (endX, Mould_pos[1]),
                         (0, 255, 255), 2)

                # Add point to tracking deque for the first vessel
                if vessel == vessels[0]:  # Track only the first detected vessel
                    self.pts.appendleft(Mould_pos)

                    # Draw the tracking line
                    for i in range(1, len(self.pts)):
                        if self.pts[i - 1] is None or self.pts[i] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        cv2.line(output, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

                # Add vessel shape information
                shape_text = f"Shape: {vessel_info['shape']}"
                cv2.putText(output, shape_text,
                            (startX, endY + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Add detection summary
        summary_text = f"Vessels detected: {len(vessels)}"
        cv2.putText(output, summary_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output, thresh


# Example usage
if __name__ == "__main__":
    print("[INFO] Initializing Vessel and Mould Level Detector...")
    detector = VesselMouldDetector()

    # Open video capture
    print("[INFO] Starting video capture...")
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

    if not cap.isOpened():
        print("[ERROR] Could not open video capture")
        exit()

    print("[INFO] Press 'q' or 'ESC' to quit")
    print("[INFO] The system will detect vessels (Liquid_Metal_Levels, cups, bowls) and measure Mould levels")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream")
            break

        # Process the frame
        output, thresh = detector.process_frame(frame)

        # Show the results
        cv2.imshow("Original", frame)
        cv2.imshow("Vessel & Mould Level Detection", output)

        # Only show threshold if it has content
        if thresh is not None and thresh.any():
            cv2.imshow("Threshold", thresh)

        # Exit on 'q' or 'ESC' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC key
            break

    print("[INFO] Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed successfully")
import cv2
import numpy as np
import imutils
from collections import deque
import os
import urllib.request


class VesselMouldDetector:
    def __init__(self):
        # Define Mould color range in HSV
        self.lower_Mould = np.array([100, 50, 50])  # Blue-ish color for Mould
        self.upper_Mould = np.array([140, 255, 255])

        # For tracking Mould level points
        self.pts = deque(maxlen=64)

        # COCO class labels that MobileNet SSD was trained on
        self.CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                        "Liquid_Metal_Level", "bus", "car", "cat", "chair", "cow", "diningtable",
                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]

        # Target vessel classes we want to detect
        self.TARGET_VESSELS = ['Liquid_Metal_Level', 'cup', 'bowl', 'wine glass', 'vase']

        # Colors for drawing bounding boxes (BGR format)
        self.COLORS = np.random.uniform(0, 255, size=(len(self.CLASSES), 3))

        # Initialize the neural network
        self.net = None
        self.load_mobilenet_ssd()

    def load_mobilenet_ssd(self):
        """Load MobileNet SSD model for object detection"""
        try:
            # Try to load local model files
            prototxt_path = "MobileNetSSD_deploy.prototxt"
            model_path = "MobileNetSSD_deploy.caffemodel"

            if not os.path.exists(prototxt_path) or not os.path.exists(model_path):
                print("[INFO] Downloading MobileNet SSD model files...")
                self.download_mobilenet_ssd()

            print("[INFO] Loading MobileNet SSD model...")
            self.net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            print("[INFO] Model loaded successfully!")

        except Exception as e:
            print(f"[ERROR] Failed to load MobileNet SSD model: {e}")
            print("[INFO] Falling back to basic contour detection...")
            self.net = None

    def download_mobilenet_ssd(self):
        """Download MobileNet SSD model files"""
        # URLs for MobileNet SSD model files
        prototxt_url = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
        model_url = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"

        try:
            # Download prototxt file
            print("[INFO] Downloading prototxt file...")
            urllib.request.urlretrieve(prototxt_url, "MobileNetSSD_deploy.prototxt")

            # Note: The model file is large and requires manual download
            print("[WARNING] Please manually download the MobileNetSSD_deploy.caffemodel file")
            print("[WARNING] from: https://github.com/chuanqi305/MobileNet-SSD")

        except Exception as e:
            print(f"[ERROR] Failed to download model files: {e}")

    def detect_vessels(self, image):
        """Detect vessels (Liquid_Metal_Levels, cups, bowls, etc.) in the image"""
        if self.net is None:
            return []

        (h, w) = image.shape[:2]

        # Create blob from image
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,
                                     (300, 300), 127.5)

        # Pass blob through network
        self.net.setInput(blob)
        detections = self.net.forward()

        vessels = []

        # Loop over detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            # Filter weak detections
            if confidence > 0.3:
                idx = int(detections[0, 0, i, 1])

                # Check if detected object is a vessel type
                if idx < len(self.CLASSES):
                    class_name = self.CLASSES[idx]

                    # Check if it's a vessel we're interested in
                    if class_name in ["Liquid_Metal_Level", "cup", "bowl"] or "table" in class_name.lower():
                        # Extract bounding box
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        vessels.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': (startX, startY, endX, endY),
                            'center': ((startX + endX) // 2, (startY + endY) // 2)
                        })

        return vessels

    def detect_Liquid_Metal_Level(self, image):
        """Detect Mould Liquid_Metal_Level in image"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Apply threshold
        _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (presumably the Liquid_Metal_Level)
        max_area = 0
        Liquid_Metal_Level_contour = None
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                Liquid_Metal_Level_contour = contour

        return Liquid_Metal_Level_contour, thresh

    def detect_Mould_level_in_vessel(self, image, vessel_bbox):
        """Detect Mould level within a specific vessel"""
        startX, startY, endX, endY = vessel_bbox

        # Extract vessel region
        vessel_roi = image[startY:endY, startX:endX]

        if vessel_roi.size == 0:
            return None, None

        # Convert to HSV and detect Mould
        hsv = cv2.cvtColor(vessel_roi, cv2.COLOR_BGR2HSV)

        # Expand Mould detection range for better detection
        lower_Mould = np.array([100, 30, 30])  # More permissive blue range
        upper_Mould = np.array([140, 255, 255])

        Mould_mask = cv2.inRange(hsv, lower_Mould, upper_Mould)

        # Also detect darker Mould/liquid
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 80])
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

        # Combine masks
        combined_mask = cv2.bitwise_or(Mould_mask, dark_mask)

        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        # Find contours in the Mould mask
        Mould_contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(Mould_contours) > 0:
            # Find the largest Mould contour
            Mould_contour = max(Mould_contours, key=cv2.contourArea)

            if cv2.contourArea(Mould_contour) > 50:  # Minimum area threshold
                # Find the topmost point of the Mould contour
                topmost = tuple(Mould_contour[Mould_contour[:, :, 1].argmin()][0])

                # Convert back to original image coordinates
                Mould_level_x = startX + topmost[0]
                Mould_level_y = startY + topmost[1]

                # Calculate Mould level percentage
                vessel_height = endY - startY
                Mould_height = endY - Mould_level_y
                Mould_percentage = (Mould_height / vessel_height) * 100 if vessel_height > 0 else 0

                return {
                    'position': (Mould_level_x, Mould_level_y),
                    'percentage': Mould_percentage,
                    'vessel_height': vessel_height,
                    'Mould_height': Mould_height
                }, combined_mask

        return None, combined_mask

    def classify_vessel_type(self, vessel_class, bbox):
        """Classify vessel type and provide specific detection parameters"""
        startX, startY, endX, endY = bbox
        width = endX - startX
        height = endY - startY
        aspect_ratio = height / width if width > 0 else 0

        vessel_info = {
            'type': vessel_class,
            'shape': 'unknown',
            'expected_Mould_region': 'bottom'
        }

        if vessel_class == 'Liquid_Metal_Level':
            if aspect_ratio > 2.0:
                vessel_info['shape'] = 'tall_Liquid_Metal_Level'
            elif aspect_ratio > 1.5:
                vessel_info['shape'] = 'standard_Liquid_Metal_Level'
            else:
                vessel_info['shape'] = 'wide_Liquid_Metal_Level'
            vessel_info['expected_Mould_region'] = 'bottom_to_neck'

        elif vessel_class == 'cup':
            vessel_info['shape'] = 'cylindrical'
            vessel_info['expected_Mould_region'] = 'bottom_portion'

        elif 'bowl' in vessel_class:
            vessel_info['shape'] = 'wide_shallow'
            vessel_info['expected_Mould_region'] = 'center_bottom'

        elif 'table' in vessel_class:
            vessel_info['shape'] = 'surface'
            vessel_info['expected_Mould_region'] = 'on_surface'

        return vessel_info

    def process_frame(self, frame):
        """Process a video frame to detect vessels and Mould levels"""
        # Make a copy for drawing
        output = frame.copy()

        # Detect vessels using MobileNet SSD
        vessels = self.detect_vessels(frame)

        # If no vessels detected with neural network, fall back to contour detection
        if not vessels:
            Liquid_Metal_Level_contour, thresh = self.detect_Liquid_Metal_Level(frame)
            if Liquid_Metal_Level_contour is not None:
                # Get bounding box from contour
                x, y, w, h = cv2.boundingRect(Liquid_Metal_Level_contour)
                vessels = [{
                    'class': 'Liquid_Metal_Level',
                    'confidence': 0.8,
                    'bbox': (x, y, x + w, y + h),
                    'center': (x + w // 2, y + h // 2)
                }]
        else:
            thresh = np.zeros(frame.shape[:2], dtype=np.uint8)

        # Process each detected vessel
        for vessel in vessels:
            vessel_class = vessel['class']
            confidence = vessel['confidence']
            bbox = vessel['bbox']
            startX, startY, endX, endY = bbox

            # Classify vessel type for better Mould detection
            vessel_info = self.classify_vessel_type(vessel_class, bbox)

            # Draw vessel bounding box
            color = [int(c) for c in self.COLORS[self.CLASSES.index(vessel_class)]]
            cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)

            # Add vessel label
            label = f"{vessel_class}: {confidence:.2f}"
            cv2.putText(output, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Detect Mould level in this vessel
            Mould_info, Mould_mask = self.detect_Mould_level_in_vessel(frame, bbox)

            if Mould_info is not None:
                Mould_pos = Mould_info['position']
                Mould_percentage = Mould_info['percentage']

                # Draw Mould level point
                cv2.circle(output, Mould_pos, 5, (255, 0, 0), -1)

                # Add Mould level information
                Mould_text = f"Mould: {Mould_percentage:.1f}%"
                cv2.putText(output, Mould_text,
                            (startX, endY + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Draw Mould level line across vessel
                cv2.line(output, (startX, Mould_pos[1]), (endX, Mould_pos[1]),
                         (0, 255, 255), 2)

                # Add point to tracking deque for the first vessel
                if vessel == vessels[0]:  # Track only the first detected vessel
                    self.pts.appendleft(Mould_pos)

                    # Draw the tracking line
                    for i in range(1, len(self.pts)):
                        if self.pts[i - 1] is None or self.pts[i] is None:
                            continue
                        thickness = int(np.sqrt(64 / float(i + 1)) * 2.5)
                        cv2.line(output, self.pts[i - 1], self.pts[i], (0, 0, 255), thickness)

                # Add vessel shape information
                shape_text = f"Shape: {vessel_info['shape']}"
                cv2.putText(output, shape_text,
                            (startX, endY + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Add detection summary
        summary_text = f"Vessels detected: {len(vessels)}"
        cv2.putText(output, summary_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return output, thresh


# Example usage
if __name__ == "__main__":
    print("[INFO] Initializing Vessel and Mould Level Detector...")
    detector = VesselMouldDetector()

    # Open video capture
    print("[INFO] Starting video capture...")
    cap = cv2.VideoCapture(0)  # Use 0 for webcam or provide video file path

    if not cap.isOpened():
        print("[ERROR] Could not open video capture")
        exit()

    print("[INFO] Press 'q' or 'ESC' to quit")
    print("[INFO] The system will detect vessels (Liquid_Metal_Levels, cups, bowls) and measure Mould levels")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream")
            break

        # Process the frame
        output, thresh = detector.process_frame(frame)

        # Show the results
        cv2.imshow("Original", frame)
        cv2.imshow("Vessel & Mould Level Detection", output)

        # Only show threshold if it has content
        if thresh is not None and thresh.any():
            cv2.imshow("Threshold", thresh)

        # Exit on 'q' or 'ESC' key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is ESC key
            break

    print("[INFO] Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Application closed successfully")
