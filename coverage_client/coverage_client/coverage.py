#! /usr/bin/env python3
# Copyright 2023 Open Navigation LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import time

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Point32, Polygon
from lifecycle_msgs.srv import GetState
from opennav_coverage_msgs.action import ComputeCoveragePath
from opennav_coverage_msgs.msg import Coordinates, Coordinate
from nav_msgs.msg import Path
import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
import pdb

from coverage_client.robot_navigator import BasicNavigator, TaskResult

class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3


class CoverageTester(Node):

    def __init__(self):
        super().__init__(node_name='coverage_tester')
        self.goal_handle = None
        self.result_future = None
        self.status = None
        self.feedback = None
        self.poses = None

        self.coverage_client = ActionClient(self, ComputeCoveragePath,
                                            'compute_coverage_path')

        self.publisher_ = self.create_publisher(Path, 'coverage_path', 10)

    def destroy_node(self):
        self.coverage_client.destroy()
        super().destroy_node()

    def toPolygon(self, field):
        poly = Coordinates()
        for coord in field:
            pt = Coordinate()
            pt.axis1 = coord[0]
            pt.axis2 = coord[1]
            poly.coordinates.append(pt)
        return poly

    def getCoverage(self, field):
        """Send a `NavToPose` action request."""
        print("Waiting for 'Coverage' action server")
        while not self.coverage_client.wait_for_server(timeout_sec=1.0):
            print('"Coverage" action server not available, waiting...')

        goal_msg = ComputeCoveragePath.Goal()
        goal_msg.frame_id = 'map'
        goal_msg.polygons.append(self.toPolygon(field))
        print('Navigating to with field of size: ' + str(len(field)) + '...')
        send_goal_future = self.coverage_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            print('Navigate Coverage request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            #pdb.set_trace()
            self.poses = self.result_future.result().result.nav_path.poses
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                print(f'Task with failed with status code: {self.status}')
                return True
            else:
                print("Publishing..")
                self.publisher_.publish(self.result_future.result().result.nav_path)
        else:
            # Timed out, still processing, not complete yet
            return False

        print('Task succeeded!')
        return True


    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN



def main():
    rclpy.init()

    coverage = CoverageTester()

    # Some example field
    field = [[0.65, 4.4], [-2.05, 1.55], [-2.2, -2.35], [-0.7, -3.55], [0.65, -2.7], [0.65, 0.9], [2.85, 4.45], [0.65, 4.4]] #[[3.35, 5.15], [-3.7, 3.25], [-1.6, -5.2], [3.35, 5.15]] #[[0.0, 0.0], [0.0, 5.0], [5.0, 5.0], [5.0, 0.0], [0.0, 0.0]]
    coverage.getCoverage(field)

    i = 0
    while not coverage.isTaskComplete():
        # Do something with the feedback
        i = i + 1
        time.sleep(1)

    # Do something depending on the return code
    result = coverage.getResult()
    if result == TaskResult.SUCCEEDED:
        print('Goal succeeded!')
    elif result == TaskResult.CANCELED:
        print('Goal was canceled!')
    elif result == TaskResult.FAILED:
        print('Goal failed!')
    else:
        print('Goal has an invalid return status!')

    navigator = BasicNavigator()
    #navigator.waitUntilNav2Active(localizer='robot_localization')

    for p in coverage.poses:
        p.header.frame_id = 'map'

    #navigator.goToPose(coverage.poses[0])
    navigator.goThroughPoses(coverage.poses)

    i = 0
    while not navigator.isTaskComplete():
        i = i + 1
        time.sleep(1)

    result = navigator.getResult()
    if result.value == TaskResult.SUCCEEDED.value:
        print('Goal succeeded!')
    elif result.value == TaskResult.CANCELED.value:
        print('Goal was canceled!')
    elif result.value == TaskResult.FAILED.value:
        print('Goal failed!')
    else:
        pdb.set_trace()
        print('Goal has an invalid return status!')

    navigator.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()
