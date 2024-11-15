"""Detects COCO objects in image and publishes in ROS2.

Subscribes to /image and publishes Detection2DArray message on topic /detected_objects.
Also publishes (by default) annotated image with bounding boxes on /annotated_image.
Uses PyTorch and FasterRCNN_MobileNet model from torchvision.
Bounding Boxes use image convention, ie center.y = 0 means top of image.
"""

import collections
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D, ObjectHypothesis, ObjectHypothesisWithPose
from vision_msgs.msg import Detection2D, Detection2DArray
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
from torchvision.models import detection as detection_model
from torchvision.utils import draw_bounding_boxes
import pdb

from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import image_geometry
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import cv2
import time
import message_filters
import yaml
import sensor_msgs_py.point_cloud2 as pc2
import tf_transformations
from geometry_msgs.msg import Point, PoseWithCovariance, Pose
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from visualization_msgs.msg import MarkerArray, Marker

Detection = collections.namedtuple("Detection", "label, bbox, score")

class CocoDetectorNode(Node):
    """Detects COCO objects in image and publishes on ROS2.

    Subscribes to /image and publishes Detection2DArray on /detected_objects.
    Also publishes augmented image with bounding boxes on /annotated_image.
    """

    # pylint: disable=R0902 disable too many instance variables warning for this class
    def __init__(self):
        super().__init__("coco_detector_node")
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('detection_threshold', 0.1)
        self.declare_parameter('publish_annotated_image', True)
        self.device = self.get_parameter('device').get_parameter_value().string_value
        self.detection_threshold = \
            self.get_parameter('detection_threshold').get_parameter_value().double_value
        self.cam = []   
        '''    
        self.subscription = self.create_subscription(
            Image,
            "/go2_camera/color/image",
            self.listener_callback,
            10)
        '''
        
        
        
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        self.detected_objects_publisher = \
            self.create_publisher(Detection2DArray, "detected_objects", 10)
        if self.get_parameter('publish_annotated_image').get_parameter_value().bool_value:
            self.annotated_image_publisher = \
                self.create_publisher(Image, "annotated_image", 10)
        else:
            self.annotated_image_publisher = None
        self.bridge = CvBridge()

        #self.class_labels = \
            #detection_model.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT.meta["categories"]
        #self.model.eval()
        
        self.model = YOLO("custom_yolov8x.pt")
        
        
        self.get_logger().info("Node has started.")
        
        self.create_subscription(
            CameraInfo,
            "/robot0/camera_info",
            self.camera_info_callback,
            qos_profile)


        #camera_info_msg = self.yaml_to_CameraInfo("/root/ost.yaml")
        
        """
        imageR = message_filters.Subscriber(self, Image, "/go2_camera/color/image")
        pointcloudR = message_filters.Subscriber(self, PointCloud2, "/point_cloud2")
        ts = message_filters.ApproximateTimeSynchronizer([pointcloudR, imageR], queue_size=10, slop=0.5)
        ts.registerCallback(self.fusion_callback)
        """
        
        self.pointcloudR = []
        self.create_subscription(
            Image,
            "/robot0/front_cam/rgb",
            self.fusion_callback,
            qos_profile)
        self.create_subscription(
            PointCloud2,
            "/robot0/point_cloud2",
            self.point_callback,
            qos_profile)
        
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        self.marker_publisher = self.create_publisher(MarkerArray, "visualization_marker_array", 10)

    def camera_info_callback(self, msg):

        camera_info_msg = msg        

        if not self.cam:
            self.cam = image_geometry.PinholeCameraModel()
            self.cam.fromCameraInfo(camera_info_msg)
        

    def mobilenet_to_ros2(self, detection, header):
        """Converts a Detection tuple(label, bbox, score) to a ROS2 Detection2D message."""

        detection2d = Detection2D()
        detection2d.header = header
        object_hypothesis_with_pose = ObjectHypothesisWithPose()
        object_hypothesis = ObjectHypothesis()
        object_hypothesis.class_id = self.class_labels[detection.label]
        object_hypothesis.score = detection.score.detach().item()
        object_hypothesis_with_pose.hypothesis = object_hypothesis
        detection2d.results.append(object_hypothesis_with_pose)
        bounding_box = BoundingBox2D()
        bounding_box.center.position.x = float((detection.bbox[0] + detection.bbox[2]) / 2)
        bounding_box.center.position.y = float((detection.bbox[1] + detection.bbox[3]) / 2)
        bounding_box.center.theta = 0.0
        bounding_box.size_x = float(2 * (bounding_box.center.position.x - detection.bbox[0]))
        bounding_box.size_y = float(2 * (bounding_box.center.position.y - detection.bbox[1]))
        detection2d.bbox = bounding_box
        return detection2d

    def publish_annotated_image(self, filtered_detections, header, image):
        """Draws the bounding boxes on the image and publishes to /annotated_image"""

        if len(filtered_detections) > 0:
            pred_boxes = torch.stack([detection.bbox for detection in filtered_detections])
            pred_labels = [self.class_labels[detection.label] for detection in filtered_detections]
            annotated_image = draw_bounding_boxes(torch.tensor(image), pred_boxes,
                                                  pred_labels, colors="yellow")
        else:
            annotated_image = torch.tensor(image)
        ros2_image_msg = self.bridge.cv2_to_imgmsg(annotated_image.numpy().transpose(1, 2, 0),
                                                   encoding="rgb8")
        ros2_image_msg.header = header
        self.annotated_image_publisher.publish(ros2_image_msg)

    def listener_callback(self, msg):
        """Reads image and publishes on /detected_objects and /annotated_image."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        '''
        image = cv_image.copy().transpose((2, 0, 1))
        batch_image = np.expand_dims(image, axis=0)
        tensor_image = torch.tensor(batch_image/255.0, dtype=torch.float, device=self.device)
        mobilenet_detections = self.model(tensor_image)[0]  # pylint: disable=E1102 disable not callable warning
        filtered_detections = [Detection(label_id, box, score) for label_id, box, score in
            zip(mobilenet_detections["labels"],
            mobilenet_detections["boxes"],
            mobilenet_detections["scores"]) if score >= self.detection_threshold]
        '''
        results = self.model.predict(cv_image)
        #pdb.set_trace()
        filtered_detections = []
        image_plot = []
        for result in results:
            self.class_labels = result.names
            image_plot = result.plot()

            filtered_detections = [Detection(int(label_id), box, score) for label_id, box, score in
                zip(result.boxes.cls,
                result.boxes.xyxy,
                result.boxes.conf) if score >= self.detection_threshold]

        detection_array = Detection2DArray()
        detection_array.header = msg.header
        detection_array.detections = \
            [self.mobilenet_to_ros2(detection, msg.header) for detection in filtered_detections]
        self.detected_objects_publisher.publish(detection_array)
        if self.annotated_image_publisher is not None:
            self.publish_annotated_image([], msg.header, image_plot.copy().transpose((2, 0, 1)))

    def yaml_to_CameraInfo(self, yaml_fname):
        """
        Parse a yaml file containing camera calibration data (as produced by 
        rosrun camera_calibration cameracalibrator.py) into a 
        sensor_msgs/CameraInfo msg.
        
        Parameters
        ----------
        yaml_fname : str
            Path to yaml file containing camera calibration data
        Returns
        -------
        camera_info_msg : sensor_msgs.msg.CameraInfo
            A sensor_msgs.msg.CameraInfo message containing the camera calibration
            data
        """
        # Load data from file
        with open(yaml_fname, "r") as file_handle:
            calib_data = yaml.load(file_handle)
        # Parse
        camera_info_msg = CameraInfo()
        camera_info_msg.width = calib_data["image_width"]
        camera_info_msg.height = calib_data["image_height"]
        camera_info_msg.k = calib_data["camera_matrix"]["data"]
        camera_info_msg.d = calib_data["distortion_coefficients"]["data"]
        camera_info_msg.r = calib_data["rectification_matrix"]["data"]
        camera_info_msg.p = calib_data["projection_matrix"]["data"]
        camera_info_msg.distortion_model = calib_data["distortion_model"]
        return camera_info_msg
        
        
    
    def transform_point(self, trans, pt):
        # https://answers.ros.org/question/249433/tf2_ros-buffer-transform-pointstamped/
        quat = [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ]
        mat = tf_transformations.quaternion_matrix(quat)
        pts_in_map_np = np.dot(mat, pt.T)
        
        pts_in_map_np = pts_in_map_np.T + np.array([trans.transform.translation.x,trans.transform.translation.y,trans.transform.translation.z,0])
        
        pts_in_map_np = np.delete(pts_in_map_np, 3, axis=1)
        """
        pt_np = [pt.x, pt.y, pt.z, 1.0]
        pt_in_map_np = np.dot(mat, pt_np)

        pt_in_map = Point()
        pt_in_map.x = pt_in_map_np[0] + trans.transform.translation.x
        pt_in_map.y = pt_in_map_np[1] + trans.transform.translation.y
        pt_in_map.z = pt_in_map_np[2] + trans.transform.translation.z
        """
        
        
        return pts_in_map_np
    
    def project_points(self, transformed_points):    
        x = (self.cam.fx()*transformed_points[:,0] + self.cam.Tx()) / transformed_points[:,2] + self.cam.cx()
        y = (self.cam.fy()*transformed_points[:,1] + self.cam.Ty()) / transformed_points[:,2] + self.cam.cy()
        
                
        return np.array([x,y]).T
        
                               
    def point_callback(self, pointcloudR):
        self.pointcloudR = pointcloudR
        
    def closest_node(self, node, nodes):
        nodes = np.asarray(nodes)

        try:
            deltas = nodes - node
            dist_2 = np.einsum('ij,ij->i', deltas, deltas)

        
            np.argpartition(dist_2, min(dist_2.size-1,4))[:4]
        except:
            pdb.set_trace()

        return np.argpartition(dist_2, min(dist_2.size-1,4))[:4]
                        
    def fusion_callback(self, imageR):
        #pdb.set_trace()
        
        time_start = time.time()
        if not self.pointcloudR or not self.cam:
            return
        try:
            t = self.tf_buffer.lookup_transform(
                "robot0/front_camera",
                "odom",
                rclpy.time.Time())
                
            toWorld = self.tf_buffer.lookup_transform(
                "map",
                "robot0/front_camera",
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform: {ex}')
            return
        
        new_points = []

        projected_points = []
        projected_pixels = []
        
        cv_image1 = self.bridge.imgmsg_to_cv2(imageR, desired_encoding="rgb8")
        cv_image = cv_image1.copy()
        self.cam.rectifyImage(cv_image1, cv_image)
        
        results = self.model.predict(cv_image)
        #pdb.set_trace()
        filtered_detections = []
        image_plot = []
        for result in results:
            self.class_labels = result.names
            image_plot = result.plot()

            filtered_detections = [Detection(int(label_id), box, score) for label_id, box, score in
                zip(result.boxes.cls,
                result.boxes.xyxy,
                result.boxes.conf) if score >= self.detection_threshold]

        detection_array = Detection2DArray()
        detection_array.header = imageR.header
        detection_array.detections = \
            [self.mobilenet_to_ros2(detection, imageR.header) for detection in filtered_detections]
        
        
        
        time_taken = time.time() - time_start
        self.get_logger().info(
                f'Time taken1: {time_taken}')
        # walk through list of points and transform each point one by one
        middle_point = []
        
        vals = pc2.read_points_numpy(self.pointcloudR, field_names=('x', 'y', 'z'))
        vals = np.hstack((vals, np.atleast_2d(np.ones(vals.shape[0])).T))
        transformed_points = self.transform_point(t, vals)

        transformed_points = transformed_points[transformed_points[:,2] > 0]
        
        transformed_pixels = self.project_points(transformed_points)
        

        circles = []
        circles_color = []
        camera_points = []
        #for x, y, z, intensity in pc2.read_points(self.pointcloudR, field_names=('x', 'y', 'z', 'intensity')):
        for pixel_idx,pixel in enumerate(transformed_pixels):
        

            #pt = Point()
            #pt.x, pt.y, pt.z = float(x), float(y), float(z)
            #new_pt = self.transform_point(t, pt)
            #new_points.append((new_pt.x, new_pt.y, new_pt.z))
            new_pt = Point()
            
            tr_point = transformed_points[pixel_idx]
            new_pt.x, new_pt.y, new_pt.z = float(tr_point[0]), float(tr_point[1]), float(tr_point[2])
            
            if new_pt.z > 0:
                #pixel = self.cam.project3dToPixel([new_pt.x, new_pt.y, new_pt.z])
              
                if (pixel[0] > 0 and pixel[0] < cv_image.shape[1]) and (pixel[1] > 0 and pixel[1] < cv_image.shape[0]):
                    projected_pixels.append(pixel)
                    projected_points.append(new_pt.z)
                    camera_points.append([new_pt.x, new_pt.y])
                    
                    #if pixel[0] > cv_image.shape[1]/2 -10 and pixel[0] < cv_image.shape[1]/2 +10 and pixel[1] > cv_image.shape[0]/2 -10 and pixel[1] < cv_image.shape[0]/2 +10:
                    #    middle_point = [pixel, new_pt]
                        

                    circles.append((int(pixel[0]),int(pixel[1])))
                    circles_color.append(min(255,max(0,int(new_pt.z*100))))
                    #color = tuple(cv2.applyColorMap(np.array([min(255,max(0,int(new_pt.z*100)))],dtype=np.uint8), cv2.COLORMAP_JET)[0][0].tolist())
                    
                    #pdb.set_trace()
                    #image_plot = cv2.circle(image_plot, (int(pixel[0]),int(pixel[1])), radius=1, color=color, thickness=-1)
                    
        #cv_image = cv2.circle(cv_image, (int(cv_image.shape[0]-1),int(cv_image.shape[1]-1)), radius=100, color=(255,255,255), thickness=-1)
        #cv_image = cv2.circle(cv_image, (int(0)+1000,int(0)), radius=100, color=(255,255,255), thickness=-1)
        #print(projected_points)
        #pdb.set_trace()    
        #print(projected_pixels)  
        #cv2.imwrite("hello.jpg", cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
        #quit()  
        
        circles_color_trans = cv2.applyColorMap(np.array(circles_color,dtype=np.uint8), cv2.COLORMAP_JET)
        #pdb.set_trace()        
        for c_idx in range(len(circles)):
            image_plot = cv2.circle(image_plot, circles[c_idx], radius=1, color=circles_color_trans[c_idx][0].tolist(), thickness=-1)
        
        time_taken = time.time() - time_start
        self.get_logger().info(
                f'Time taken2: {time_taken}')
                
        points_to_world = []
        for det in detection_array.detections:
            det_point = [int(det.bbox.center.position.x), int(det.bbox.center.position.y+det.bbox.size_y/2)]
            if not projected_pixels:
                break
            point_num = self.closest_node(det_point, projected_pixels)
            distance = float(np.mean(np.array(projected_points)[point_num]))
            #distance_vec = self.cam.projectPixelTo3dRay([int(det.bbox.center.position.x), int(det.bbox.center.position.y-det.bbox.size_y/2)])
            image_plot = cv2.putText(image_plot, str(distance), det_point, cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (255,0,0), 2, cv2.LINE_AA)
            

            points_to_world.append(np.array(self.cam.projectPixelTo3dRay(det_point))*distance)
        
        points_to_world = np.array(points_to_world)
        new_transformed_points = []
        if points_to_world.size > 0:
            points_to_world = np.hstack((points_to_world, np.atleast_2d(np.ones(points_to_world.shape[0])).T))
            new_transformed_points = self.transform_point(toWorld, points_to_world)
        
        markerArray = MarkerArray()
        
        #pdb.set_trace()
        for newtp_idx,newtp in enumerate(new_transformed_points):
        
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            
            if detection_array.detections[newtp_idx].results[0].hypothesis.class_id == "box":
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 1.0
            
            marker.color.a = 1.0
            
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = float(newtp[0])
            marker.pose.position.y = float(newtp[1]) 
            marker.pose.position.z = float(newtp[2])
            marker.id = newtp_idx
            marker.ns = "detected_objects"
            
            markerArray.markers.append(marker) 
            
            
            detection_array.detections[newtp_idx].results[0].pose.pose.position.x = float(newtp[0])
            detection_array.detections[newtp_idx].results[0].pose.pose.position.y = float(newtp[1])
            detection_array.detections[newtp_idx].results[0].pose.pose.position.z = float(newtp[2])
            detection_array.detections[newtp_idx].header.frame_id = "map"
            
        self.marker_publisher.publish(markerArray)
        
        if self.annotated_image_publisher is not None:
            self.publish_annotated_image([], imageR.header, image_plot.transpose((2, 0, 1)))
        
        
        self.detected_objects_publisher.publish(detection_array)
        
        time_taken = time.time() - time_start
        self.get_logger().info(
                f'Time taken3: {time_taken}')
rclpy.init()
coco_detector_node = CocoDetectorNode()
rclpy.spin(coco_detector_node)
coco_detector_node.destroy_node()
rclpy.shutdown()
