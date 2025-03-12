# import torch
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from ultralytics import YOLO
# from torchvision.transforms import Compose, Resize, ToTensor, Normalize
# import torch.hub
# from PIL import Image
# import csv

# # MiDaS  l'estimation de profondeur
# def load_midas():
#     model_type = "DPT_Large"
#     midas = torch.hub.load("intel-isl/MiDaS", model_type)
#     midas.eval()
#     transform = Compose([
#         Resize(384 if model_type == "MiDaS_small" else 512),
#         ToTensor(),
#         Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return midas, transform

# # Détecter les objets avec YOLOv8
# def detect_objects(image, model):
#     results = model(image)
#     detections = []
#     for result in results:
#         for box in result.boxes:
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             obj_name = model.names[int(box.cls)]
#             confidence = float(box.conf)
#             detections.append((obj_name, confidence, x1, y1, x2, y2))
#     return detections

# # Estimer la profondeur avec MiDaS
# def estimate_depth(image, midas, transform):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image_pil = Image.fromarray(image_rgb)
#     img_tensor = transform(image_pil).unsqueeze(0)

#     with torch.no_grad():
#         depth_map = midas(img_tensor)

#     depth_map = depth_map.squeeze().cpu().numpy()
#     depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
#     return depth_map

# # Calculer la profondeur moyenne dans une région d'intérêt (ROI)
# def compute_average_depth(depth_map, x1, y1, x2, y2):
#     roi = depth_map[y1:y2, x1:x2]
#     return np.median(roi)

# # Convertir la profondeur en distance
# def convert_depth_to_distance(avg_depth, scale_factor=30.4255225):
#     return scale_factor / (avg_depth + 1e-6)  # Éviter la division par zéro

# # Charger les modèles
# midas, transform = load_midas()
# yolo_model = YOLO(r"C:\Users\Lenovo\Desktop\Go1\best.pt")  # Chemin vers le modèle YOLOv8

# # Ouvrir la caméra
# cap = cv2.VideoCapture(0)  # Utiliser la webcam (remplacez 0 par le numéro de la caméra si nécessaire)

# if not cap.isOpened():
#     raise RuntimeError("Erreur : Impossible d'ouvrir la caméra.")

# # Boucle principale pour le traitement en temps réel
# try:
#     while True:
#         # Capturer une frame
#         ret, frame = cap.read()
#         if not ret:
#             print("Erreur : Impossible de capturer la frame.")
#             break

#         # Estimer la profondeur
#         depth_map = estimate_depth(frame, midas, transform)

#         # Détecter les objets
#         detections = detect_objects(frame, yolo_model)

#         # Afficher les résultats
#         depth_overlay = frame.copy()
#         for obj_name, confidence, x1, y1, x2, y2 in detections:
#             avg_depth = compute_average_depth(depth_map, x1, y1, x2, y2)
#             distance = convert_depth_to_distance(avg_depth, scale_factor=30.3456789)
#             cv2.rectangle(depth_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             cv2.putText(depth_overlay, f"{obj_name}: {distance:.2f}m", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Afficher la frame avec les détections
#         cv2.imshow("YOLOv8 Object Detection with Distance", depth_overlay)

#         # Afficher la carte de profondeur
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_MAGMA)
#         cv2.imshow("Depth Estimation", depth_colormap)

#         # Quitter avec la touche 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("Fermeture de l'application.")
#             break
# finally:
#     # Libérer les ressources
#     cap.release()
#     cv2.destroyAllWindows()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch.hub
from PIL import Image as PILImage



# MiDaS for depth estimation
def load_midas():
    model_type = "DPT_Large"
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    midas.eval()
    transform = Compose([
        Resize(384 if model_type == "MiDaS_small" else 512),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return midas, transform

# Detect objects with YOLOv8
def detect_objects(image, model):
    results = model(image)
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            obj_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            detections.append((obj_name, confidence, x1, y1, x2, y2))
    return detections

# Estimate depth with MiDaS
def estimate_depth(image, midas, transform):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_pil = PILImage.fromarray(image_rgb)
    img_tensor = transform(image_pil).unsqueeze(0)

    with torch.no_grad():
        depth_map = midas(img_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]))
    return depth_map

# Compute average depth in a region of interest (ROI)
def compute_average_depth(depth_map, x1, y1, x2, y2):
    roi = depth_map[y1:y2, x1:x2]
    return np.median(roi)

# Convert depth to distance
def convert_depth_to_distance(avg_depth, scale_factor=30.4255225):
    return scale_factor / (avg_depth + 1e-6)  # Avoid division by zero

# ROS 2 Node for Object Detection and Depth Estimation
class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        self.bridge = CvBridge()

        # Subscribe to the RealSense camera's color image topic
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',  # Adjust the topic name if necessary
            self.image_callback,
            10)
        self.subscription  # Prevent unused variable warning

        # Load models
        self.midas, self.transform = load_midas()
        self.yolo_model = YOLO("/path/to/best.pt")  # Update the path to your YOLO model

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Failed to convert image: {e}")
            return

        # Estimate depth
        depth_map = estimate_depth(frame, self.midas, self.transform)

        # Detect objects
        detections = detect_objects(frame, self.yolo_model)

        # Display results
        depth_overlay = frame.copy()
        for obj_name, confidence, x1, y1, x2, y2 in detections:
            avg_depth = compute_average_depth(depth_map, x1, y1, x2, y2)
            distance = convert_depth_to_distance(avg_depth, scale_factor=30.3456789)
            cv2.rectangle(depth_overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(depth_overlay, f"{obj_name}: {distance:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("YOLOv8 Object Detection with Distance", depth_overlay)

        # Display the depth map
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.03), cv2.COLORMAP_MAGMA)
        cv2.imshow("Depth Estimation", depth_colormap)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Closing application.")
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()