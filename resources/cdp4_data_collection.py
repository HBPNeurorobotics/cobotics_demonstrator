import os
import rospy
import numpy as np
import cameratransform as ct
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from gazebo_msgs.msg import ModelState, ModelStates
from cv_bridge import CvBridge, CvBridgeError
from gazebo_msgs.srv import GetModelState, SetModelState, GetWorldProperties, SpawnEntity,\
    DeleteModel, SetVisualProperties, SetVisualPropertiesRequest
from tf.transformations import quaternion_from_euler, euler_from_quaternion


class CDP4DataCollection:

    def __init__(self, camera_params):

        self.bridge = CvBridge()
        for camera_id in range(camera_params['n_cameras']):
            setattr(self, 'last_image_%02i' % (camera_id,), [None])
        self.last_model_states = ModelStates()
        self.spawned_objects = []

        # Camera IDs and corresponding variables holding images
        self.camera_type = camera_params['camera_type']
        self.camera_id_dict = {'%02i' % (camera_id,): getattr(self, 'last_image_%02i'\
            % (camera_id,)) for camera_id in range(camera_params['n_cameras'])}
        
        # ROS Subscribers, Publishers, Services
        for camera_id in range(camera_params['n_cameras']):
            img_callback = self.__image_callback_wrapper(camera_id)
            setattr(self, '__image_%02i_sub' % (camera_id,), rospy.Subscriber(
                '/robot/camera_%s_%02i' % (self.camera_type, camera_id),
                Image, img_callback, queue_size=1))

        self.__set_model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState,
            queue_size=1)
        
        rospy.wait_for_service('gazebo/get_model_state', 10.0)
        self.__get_model_state_srv = rospy.ServiceProxy('gazebo/get_model_state', GetModelState)


        # 3D voxel to 2D camera pixel projection parameters (useful for segmentation dataset)
        self.cam_transform = []
        for camera_id in range(camera_params['n_cameras']):
            cam_pose = self.get_object_pose('camera_%02i' % (camera_id))
            cam_pos = cam_pose.position
            cam_ori = cam_pose.orientation
            roll, pitch, yaw = euler_from_quaternion([cam_ori.x, cam_ori.y, cam_ori.z, cam_ori.w])
            (roll, pitch, yaw) = (angle_in_rad*180/np.pi for angle_in_rad in (roll, pitch, yaw))

            cam_projection = ct.RectilinearProjection(
                focallength_px=camera_params['focal_length_px'],
                image=camera_params['camera_resolution'])
            cam_orientation = ct.SpatialOrientation(
                pos_x_m=cam_pos.x, pos_y_m=cam_pos.y, elevation_m=cam_pos.z,
                roll_deg=roll, tilt_deg=90-pitch, heading_deg=90-yaw)
            cam_lens = ct.BrownLensDistortion(
                k1=0.0, k2=0.0, k3=0.0, projection=cam_projection)
            self.cam_transform.append(ct.Camera(
                projection=cam_projection, orientation=cam_orientation, lens=cam_lens))

        rospy.init_node('cdp4_data_collection')
        rospy.wait_for_service('/gazebo/get_world_properties')

        self.__physics_state = rospy.ServiceProxy('/gazebo/get_world_properties',
                                                  GetWorldProperties)

        while self.__physics_state().sim_time < 2:
            print('Waiting for simulation to be started')
            rospy.sleep(2)

        self.__path_to_models = os.getenv('HOME') + '/.gazebo/models/'

        rospy.wait_for_service('gazebo/set_model_state', 10.0)
        self.__set_model_state_srv = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)

        rospy.wait_for_service('gazebo/spawn_sdf_entity')
        self.__spawn_model_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_entity', SpawnEntity)
        
        rospy.wait_for_service('gazebo/set_visual_properties')
        self.__change_color_srv = rospy.ServiceProxy('gazebo/set_visual_properties',
                                                     SetVisualProperties)

    def __image_callback_wrapper(self, camera_id):
        """
        Returns correct image saving function

        :param camera_id: The id of the camera to be used 
        """
        last_image = getattr(self, 'last_image_%02i' % (camera_id,))
        def image_callback(msg):
            """
            Saves the last published image to last_image

            :param msg: The ROS message
            """
            try:
                timestamp = (msg.header.stamp.secs, msg.header.stamp.nsecs)
                cv2_img = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
            except CvBridgeError, e:
                print e
            else:
                last_image[0] = (cv2_img, timestamp)
        return image_callback

    def add_object(self, model_name, pose, reference_frame='world'):
        """
        Spawns a new object in the environment

        :param model_name: The model name of the object to be spawned
        :param pose: The pose where the object will be spawned, relative to world coordinates
        :param reference_frame: the reference frame in which the pose will be considered
        """
        with open(self.__path_to_models + model_name + '/model.sdf', 'r') as model:
            sdf = model.read()

        while model_name in self.last_model_states.name:
            parts = model_name.split('_')
            try:
                parts[-1] = str(int(parts[-1]) + 1)
            except:
                parts.append('1')
            model_name = "_".join(parts)

        res = self.__spawn_model_srv(model_name, sdf, '', pose, reference_frame)
        rospy.loginfo(res)

    def delete_object(self, model_name):
        """
        Delete an object from the environment

        :param model_name: The model name of the object to be spawned
        """
        res = self.__delete_model_srv(model_name)
        rospy.loginfo(res)

    def set_object_color(self, model_name, color, link_name='link', visual_name='visual'):
        """
        Changes the color of objects in the scene

        :param model_name: The name of the model
        :param link_name: The name of the link whose color is going to be changed
        :param visual_name: The name of the visual whose color is going to be changed
        :param color: Standard Gazebo color (e.g., 'Gazebo/Red')
        """
        request = SetVisualPropertiesRequest(model_name, link_name, visual_name,
                                             'material:script:name', color)
        self.__change_color_srv(request)

    def get_object_pose(self, object_name, reference_frame='world'):
        """
        Gets the current pose of an object relative to the world's coordinate frame

        :param object_name: the model name of the object
        :param reference_frame: the reference frame from which the pose will be calculated
        """
        return self.__get_model_state_srv(object_name, reference_frame).pose

    def set_object_pose(self, object_name, pose, store):
        """
        Sets an object's pose in the scene

        :param object_name: the name of the object model
        :param pose: the new pose the model should be set to
        """
        if store:
            self.spawned_objects.append(object_name)
        
        msg = ModelState()

        msg.model_name = object_name
        msg.reference_frame = 'world'
        msg.pose = pose
        object_scale = self.get_object_scale(object_name)
        msg.scale.x = object_scale.x
        msg.scale.y = object_scale.y
        msg.scale.z = object_scale.z

        # publish message on ros topic
        self.__set_model_state_pub.publish(msg)

    def get_object_scale(self, object_name):
        """
        Gets the current scales of an object

        :param object_name: the model name of the object
        :param reference_frame: the reference frame from which the pose will be calculated
        """
        return self.__get_model_state_srv(object_name, 'world').scale

    def set_object_scale(self, object_name, x=None, y=None, z=None):
        """
        Modifies an object's scale along a given axis

        :param object_name: the name of the object model
        :param scale: (float) the new scale of the object
        """
        model_state = self.__get_model_state_srv(object_name, 'world')
        msg = ModelState()
        msg.model_name = object_name
        msg.reference_frame = 'world'
        msg.pose = model_state.pose
        msg.scale.x = x if x else model_state.scale.x
        msg.scale.y = y if y else model_state.scale.y
        msg.scale.z = z if z else model_state.scale.z

        # publish message on ros topic
        self.__set_model_state_pub.publish(msg)

    def move_camera(self, camera_id, x=0, y=0, z=0, roll=0, pitch=0, yaw=0):
        """
        Moves the robot camera by a desired amount of distance (meter) and angle (rad)
        """
        model_name = 'camera_%02i' % (camera_id)
        cur_pose = self.get_object_pose(model_name)
        cur_orientation = euler_from_quaternion([cur_pose.orientation.x,
                                                 cur_pose.orientation.y,
                                                 cur_pose.orientation.z,
                                                 cur_pose.orientation.w])

        orientation = quaternion_from_euler(cur_orientation[0] + roll,
                                            cur_orientation[1] + pitch,
                                            cur_orientation[2] + yaw)
        msg = ModelState()
        msg.model_name = model_name
        msg.reference_frame = 'world'
        msg.scale.x = 1
        msg.scale.y = 1
        msg.scale.z = 1
        msg.pose.position.x = cur_pose.position.x + x
        msg.pose.position.y = cur_pose.position.y + y
        msg.pose.position.z = cur_pose.position.z + z
        msg.pose.orientation.x = orientation[0]
        msg.pose.orientation.y = orientation[1]
        msg.pose.orientation.z = orientation[2]
        msg.pose.orientation.w = orientation[3]
        self.__set_model_state_srv(msg)

        # Update 3D world to 2D camera array projection state
        self.cam_transform[camera_id].pos_x_m += x
        self.cam_transform[camera_id].pos_y_m += y
        self.cam_transform[camera_id].elevation_m += z
        self.cam_transform[camera_id].roll_deg += roll*180/np.pi
        self.cam_transform[camera_id].tilt_deg -= pitch*180/np.pi
        self.cam_transform[camera_id].heading_deg -= yaw*180/np.pi
        
    def capture_image(self, camera_id, ensure_timing=False):
        """
        Captures an image with a time stamp greater than the current time. This helps us overcome
        ROS synchronization issues and ensures that we don't get images from the past

        :param camera_id: The id of the camera
        :ensure_timing: Do the time check or not (slower)?
        """
        if ensure_timing:

            # current time
            now = rospy.get_rostime() - rospy.Time(secs=0)

            # ensure that the time stamp of the image is greater than the current time by at least 0.5s
            time_step = 0.01
            while (now + rospy.Duration(0, time_step)) > rospy.Duration(
                    self.camera_id_dict['%02i' % (camera_id,)][0][1][0],
                    self.camera_id_dict['%02i' % (camera_id,)][0][1][1]):
                rospy.sleep(time_step)

        return self.camera_id_dict['%02i' % (camera_id,)][0][0]
