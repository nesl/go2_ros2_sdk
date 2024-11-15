import numpy as np
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2D, Detection2DArray
from visualization_msgs.msg import MarkerArray, Marker
import pdb
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from collections import deque


class Object_Localization(Node):

    def __init__(self):
        super().__init__("object_localization")

        print("starting object localization")
        self.object_coords = np.array([])
        self.object_all_coords = []
        self.max_buffer_coords = 10

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(Detection2DArray, "/detected_objects", self.localization_callback, 10)

        self.marker_publisher = self.create_publisher(MarkerArray, "object_localization", 10)

    def localization_callback(self, msg):
        
        
        try:
            trans = self.tf_buffer.lookup_transform(
                "map",
                "robot0/base_link",
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().info(
                f'Could not transform: {ex}')
            return 
        
        current_position = np.array([trans.transform.translation.x,trans.transform.translation.y])
        
        for detection in msg.detections:
            

            point = np.array([detection.results[0].pose.pose.position.x,detection.results[0].pose.pose.position.y])


            if np.linalg.norm(current_position-point) < 1.5:
                if self.object_coords.size > 0:

                    distances = np.sqrt(np.sum((self.object_coords-point)**2,axis=1))

                    if not np.where(distances < 0.5)[0].size:                                

                        self.object_coords = np.concatenate((self.object_coords, np.expand_dims(point,0)))
                        self.object_all_coords.append(deque(maxlen=self.max_buffer_coords))
                        self.object_all_coords[-1].append(point)
                    else:
                        ob_idx = np.argmin(distances)
                        self.object_all_coords[ob_idx].append(point)
                        try:
                             av_coords = np.average(np.array(self.object_all_coords[ob_idx]), axis=0)
                             self.object_coords[ob_idx][0] = av_coords[0]
                             self.object_coords[ob_idx][1] = av_coords[1]
                        except:
                            pdb.set_trace()
                        
                            
                else:
                    self.object_coords = np.expand_dims(np.append(self.object_coords,point), 0)
                    self.object_all_coords.append(deque(maxlen=self.max_buffer_coords))
                    self.object_all_coords[-1].append(point)
                    
          
        #Merge similar points
        if self.object_coords.shape[0] > 1:
            self_distances = np.linalg.norm(self.object_coords[np.newaxis, :, :] - self.object_coords[:, np.newaxis, :], axis=2)
            sd = self_distances[np.triu_indices(self.object_coords.shape[0], 1)]

            sd_min = np.argmin(sd)

            if sd[sd_min] < 0.5:
                similar_points = np.where(self_distances == sd[sd_min])
                self.object_all_coords[similar_points[0][0]].extend(self.object_all_coords[similar_points[1][0]])
                av_coords = np.average(np.array(self.object_all_coords[similar_points[0][0]]), axis=0)
                self.object_coords[similar_points[0][0]][0] = av_coords[0]
                self.object_coords[similar_points[0][0]][1] = av_coords[1]
                del self.object_all_coords[similar_points[1][0]]
                self.object_coords = np.delete(self.object_coords, similar_points[1][0],0)

        mask = [len(obc) >= self.max_buffer_coords for obc in self.object_all_coords]
        #print(self.object_all_coords, mask)
        self.marker_publish(self.object_coords, mask)
            
    def marker_publish(self, points, mask):
        markerArray = MarkerArray()
        
        #pdb.set_trace()
        for newtp_idx,newtp in enumerate(points):
        
            if not mask[newtp_idx]:
                continue
        
            marker = Marker()
            marker.header.frame_id = "/map"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.2
            marker.scale.y = 0.2
            marker.scale.z = 0.2
            

            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0

            #marker.color.r = 0.0
            #marker.color.g = 1.0
            #marker.color.b = 1.0
            
            marker.color.a = 1.0
            
            marker.pose.orientation.w = 1.0
            try:
                marker.pose.position.x = float(newtp[0])
                marker.pose.position.y = float(newtp[1]) 
            except:
                pdb.set_trace()
            marker.pose.position.z = float(0)#float(newtp[2])
            marker.id = newtp_idx
            marker.ns = ""
            
            markerArray.markers.append(marker) 
            
        self.marker_publisher.publish(markerArray)

def main():

    rclpy.init()
    localization_node = Object_Localization()
    rclpy.spin(localization_node)
    localization_node.destroy_node()
    rclpy.shutdown()
