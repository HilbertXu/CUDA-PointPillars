import os
import rospy
import ros_numpy
import numpy as np
from sensor_msgs.msg import PointCloud2

class BagToSequence:
    def __init__(self, target_dir) -> None:

        rospy.init_node("bag_to_seq")

        self.target_dir = target_dir
        self.count = 0

        self.pcl_sub = rospy.Subscriber("/lio_sam/mapping/cloud_surf", PointCloud2, self.pcl_callback)
    

    def pcl_callback(self, msg):
        data = ros_numpy.numpify(msg)
        points=np.zeros((data.shape[0],4))
        points[:,0]=data['x']
        points[:,1]=data['y']
        points[:,2]=data['z']
        points[:,3]=data["intensity"]

        points = points.astype(np.float32)
        print(points.shape)
        points.tofile(os.path.join(self.target_dir, "{:06d}.bin".format(self.count)))
        self.count += 1


if __name__ == "__main__":
    converter = BagToSequence("/home/hilbertxu/puzek_ws/CUDA-PointPillars/seq")
    rospy.spin()

        