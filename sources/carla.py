import sys
import os
import glob

import random
import carla
import math
import time
import cv2
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree as kdtree

import settings

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


class CarEnv:

    actor_list = []
    collision_hist = []

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(180.0)

        # Load layered map for Town03 with minimum layout plus buildings and parked vehicles
        # self.world = self.client.get_world()  # Uncomment this and comment self.client.load_world() if you don't want to change the map
        self.world = self.client.load_world(
            'Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        # Toggle all buildings off
        # self.world.unload_map_layer(carla.MapLayer.Buildings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]

        for x in list(self.world.get_actors()):
            if 'vehicle' in x.type_id or 'sensor' in x.type_id:
                x.destroy()

        self.blueprint_library = self.world.get_blueprint_library()
        self.blueprint = self.blueprint_library.filter('model3')[0]

        # Autopilot vehicle
        tm = self.client.get_trafficmanager()
        tm_port = tm.get_port()

        self.auto_vehicle_list = []
        while len(self.auto_vehicle_list) < settings.NUMBER_OF_AUTO_VEHICLES:
            spawn_point = random.choice(
                self.world.get_map().get_spawn_points())
            auto_vehicle = self.world.try_spawn_actor(
                self.blueprint, spawn_point)
            if auto_vehicle is not None:
                auto_vehicle.apply_control(
                    carla.VehicleControl(throttle=0.0, steer=0.0))
                time.sleep(3)   # Wait for a car to be ready
                auto_vehicle.set_autopilot(True, tm_port)
                tm.ignore_lights_percentage(auto_vehicle, 100)
                tm.vehicle_percentage_speed_difference(auto_vehicle, 50)
                self.auto_vehicle_list.append(auto_vehicle)

        self.test = False

    def reset(self):
        self.collision_hist = []
        self.actor_list = []

        # Random maps
        # 0-9 are indexes of maps. For now, we use 0-7 for training and 8-9 for testing.
        while True:
            if self.test == False:
                index = random.randint(0, 7)
            elif self.test == True:
                index = random.randint(0, 9)
            self.map_name = 'refmap3-{}'.format(index+1)
            self.refmap = pd.read_csv('refmaps\\{}.csv'.format(self.map_name))
            self.kd_tree_map = kdtree(self.refmap.values)

            starting_points = pd.read_csv('refmaps\\starting_points.csv')
            starting_point = (
                starting_points.iloc[index, 1], starting_points.iloc[index, 2], starting_points.iloc[index, 3])  # (x,y,yaw)

            # Vehicle
            transform = carla.Transform(carla.Location(
                starting_point[0], starting_point[1], 0.5), carla.Rotation(yaw=starting_point[2]))
            self.vehicle = self.world.try_spawn_actor(
                self.blueprint, transform)
            if self.vehicle != None:
                break
        self.actor_list.append(self.vehicle)

        # Spectator
        spectator = self.world.get_spectator()
        transform = carla.Transform(carla.Location(
            starting_point[0], starting_point[1], 3), carla.Rotation(yaw=starting_point[2]))
        spectator.set_transform(transform)

        # Depth camera
        self.depth_bp = self.blueprint_library.find('sensor.camera.depth')
        self.depth_bp.set_attribute("image_size_x", f"{settings.IM_WIDTH}")
        self.depth_bp.set_attribute("image_size_y", f"{settings.IM_HEIGHT}")
        self.depth_bp.set_attribute("fov", f"110")

        depth_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.depth_cam = self.world.spawn_actor(
            self.depth_bp, depth_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.depth_cam)
        self.depth_cam.listen(lambda data: self.process_img_depth(data))

        # Segmentation camera
        self.segm_bp = self.blueprint_library.find(
            'sensor.camera.semantic_segmentation')
        self.segm_bp.set_attribute("image_size_x", f"{settings.IM_WIDTH}")
        self.segm_bp.set_attribute("image_size_y", f"{settings.IM_HEIGHT}")
        self.segm_bp.set_attribute("fov", f"110")

        segm_transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        self.segm_cam = self.world.spawn_actor(
            self.segm_bp, segm_transform, attach_to=self.vehicle, attachment_type=carla.AttachmentType.Rigid)
        self.actor_list.append(self.segm_cam)
        # This time, a color converter is applied to the image, to get the semantic segmentation view
        self.segm_cam.listen(lambda data: self.process_img_segm(data))

        # LiDAR
        self.lidar_sensor = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        self.lidar_sensor.set_attribute('points_per_second', '100000')
        self.lidar_sensor.set_attribute('channels', '32')
        self.lidar_sensor.set_attribute('range', '10000')
        self.lidar_sensor.set_attribute('upper_fov', '10')
        self.lidar_sensor.set_attribute('lower_fov', '-10')
        self.lidar_sensor.set_attribute('rotation_frequency', '60')

        transform = carla.Transform(carla.Location(x=0, z=1.8))
        self.lidar = self.world.spawn_actor(
            self.lidar_sensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.lidar)
        self.lidar.listen(lambda data: self.process_lidar(data))

        # Apply control
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))
        # Sleep to get things started and to not detect a collision when the car spawns/falls from sky.
        time.sleep(4)

        transform = carla.Transform(carla.Location(x=2.5, z=0.7))
        colsensor = self.world.get_blueprint_library().find('sensor.other.collision')
        self.colsensor = self.world.spawn_actor(
            colsensor, transform, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        while self.distance_to_obstacle_f is None:
            time.sleep(0.01)

        self.episode_start = time.time()

        # Initialize state
        alpha = self.get_tangent_angle_closest_point()
        beta = alpha
        diff_angle = beta-alpha

        distance_from_road = self.get_distance()

        side = 1    # side = self.get_side()
        last_action = [0, 0]  # last_action [throttle, steer]
        diff_v = 0
        v_kmh = 0

        xx = int(self.distance_to_obstacle_f)
        yy = int(self.distance_to_obstacle_r)
        zz = int(self.distance_to_obstacle_l)

        img = np.random.rand(settings.IM_HEIGHT, settings.IM_WIDTH, 2)
        img[:, :, 0:1] = self.segm_img
        img[:, :, 1:2] = self.depth

        self.current_state = np.array(
            [img, xx, yy, zz, diff_angle, distance_from_road, side, last_action[1], diff_v, v_kmh], dtype='object')

        return self.current_state

    #############################################################################
    def get_tangent_angle_closest_point(self):
        l = self.vehicle.get_location()
        current_location = [l.x, l.y]
        id = self.kd_tree_map.query(current_location)[1]
        if id == 0:
            bf = 0
            af = 1
        elif id == len(self.kd_tree_map.data)-1:
            bf = id-1
            af = id
        else:
            bf = id-1
            af = id+1
        xbf = self.kd_tree_map.data[bf][0]
        ybf = self.kd_tree_map.data[bf][1]
        xaf = self.kd_tree_map.data[af][0]
        yaf = self.kd_tree_map.data[af][1]
        alpha = math.atan2((yaf-ybf), (xaf-xbf))

        # Distance Percentage
        length = self.refmap.shape[0]
        self.distance_percentage = (af/length)*100

        return alpha

    def get_velocity_data(self):
        vcar = self.vehicle.get_velocity()
        vx = vcar.x
        vy = vcar.y
        beta = math.atan2(vy, vx)
        vtotal = math.sqrt(math.pow(vx, 2) + math.pow(vy, 2))
        return beta, vtotal

    def get_distance(self):
        l = self.vehicle.get_location()
        current_location = [l.x, l.y]
        dist, id = self.kd_tree_map.query(current_location, 1)
        return dist

    def get_location(self):
        l = self.vehicle.get_location()
        current_location = [l.x, l.y]
        return current_location

    def get_side(self):
        ref_map = self.get_distance()
        l = self.vehicle.get_location()
        id = self.kd_tree_map.query([l.x, l.y])[1]
        x = self.kd_tree_map.data[id][0]
        y = self.kd_tree_map.data[id][1]
        alpha = math.atan2((l.y-y), (l.x-x))
        beta = self.get_tangent_angle_closest_point()
        if beta*alpha < 0:
            if abs(beta)+abs(alpha) > math.pi:
                if beta > alpha:
                    return 1
                else:
                    return 2
            else:
                if beta > alpha:
                    return 2
                else:
                    return 1
        else:
            if beta > alpha:
                return 2
            else:
                return 1

    def get_distance_to_finish(self):
        l = self.vehicle.get_location()
        destination = (self.refmap.iloc[-1, 0], self.refmap.iloc[-1, 1])
        dist = math.sqrt(
            math.pow(destination[0]-l.x, 2) + math.pow(destination[1]-l.y, 2))
        return dist

    #############################################################################

    def collision_data(self, event):
        self.collision_hist.append(event)

    #############################################################################
    def process_lidar(self, raw):
        points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(
            points, (int(points.shape[0] / 4), 4))[:, :3]*np.array([1, -1, -1])

        lidar_f = self.lidar_line(points, 90, 2)
        lidar_r = self.lidar_line(points, 45, 2)
        lidar_l = self.lidar_line(points, 135, 2)

        CAR_LENGTH = 2.482  # Depending on which car you use

        if len(lidar_f) == 0:
            pass
        else:
            self.distance_to_obstacle_f = min(lidar_f[:, 1])-CAR_LENGTH

        if len(lidar_r) == 0:
            pass
        else:
            self.distance_to_obstacle_r = np.sqrt(
                min(lidar_r[:, 0]**2 + lidar_r[:, 1]**2))

        if len(lidar_l) == 0:
            pass
        else:
            self.distance_to_obstacle_l = np.sqrt(
                min(lidar_l[:, 0]**2 + lidar_l[:, 1]**2))

    def lidar_line(self, points, degree, width):
        angle = degree*(2*np.pi)/360
        points_l = points
        points_l = points_l[np.logical_and(
            points_l[:, 2] > -1.75, points_l[:, 2] < 1000)]  # z
        points_l = points_l[np.logical_and(np.tan(angle)*points_l[:, 0]+width*np.sqrt(1+np.tan(angle)**2) >=
                                           points_l[:, 1], np.tan(angle)*points_l[:, 0]-width*np.sqrt(1+np.tan(angle)**2) <= points_l[:, 1])]  # y
        if 180 > degree > 0:
            points_l = points_l[np.logical_and(
                points_l[:, 1] > 0, points_l[:, 1] < 1000)]  # y>0
        if 180 < degree < 360:
            points_l = points_l[np.logical_and(
                points_l[:, 1] < 0, points_l[:, 1] > -1000)]  # x
        if degree == 0 or degree == 360:
            points_l = points_l[np.logical_and(
                points_l[:, 0] > 0, points_l[:, 0] < 1000)]  # x
        if degree == 180:
            points_l = points_l[np.logical_and(
                points_l[:, 0] > -1000, points_l[:, 0] < 0)]
        return points_l

    def process_img_segm(self, img):
        i = np.array(img.raw_data)
        i2 = i.reshape((settings.IM_HEIGHT, settings.IM_WIDTH, 4))
        self.segm_img = i2[:, :, 2:3]

        if settings.SHOW_CAM:
            img.convert(color_converter=carla.ColorConverter.CityScapesPalette)
            i4 = np.array(img.raw_data)
            i5 = i4.reshape((settings.IM_HEIGHT, settings.IM_WIDTH, 4))
            i6 = i5[:, :, :3]
            cv2.imshow("segmentation", i6)
            cv2.waitKey(1)

    def process_img_depth(self, img):
        i = np.array(img.raw_data)
        i2 = i.reshape((settings.IM_HEIGHT, settings.IM_WIDTH, 4))
        i3 = i2[:, :, :3]

        R = np.array(i3[:, :, 0], dtype=np.float64)
        G = np.array(i3[:, :, 1], dtype=np.float64)
        B = np.array(i3[:, :, 2], dtype=np.float64)
        # distances in km for which the maximum value is 1.0
        self.depth = (R + G*256 + B*256*256)/(256*256*256-1)
        # self.depth = 1000 * self.depth   # in meters; shape = (IM_HEIGHT, IM_WIDTH)
        self.depth = self.depth.reshape(
            (settings.IM_HEIGHT, settings.IM_WIDTH, 1))

        # For color conversion purpose only
        if settings.SHOW_CAM:
            img.convert(color_converter=carla.ColorConverter.LogarithmicDepth)
            i = np.array(img.raw_data)
            i2 = i.reshape((settings.IM_HEIGHT, settings.IM_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imshow("depth", i3)
            cv2.waitKey(1)

    ##############################################################################################

    def step(self, action, last_action, last_velocity):

        is_finished = False

        action = [act.item() for act in action]

        # In case of the car unexpectedly stopped -- You can comment this if this is unnecessary.
        if self.test == True and action[0] == -1.0:
            action[0] = -0.75

        # Apply control
        self.vehicle.apply_control(carla.VehicleControl(
            throttle=0.5+0.5*action[0], steer=action[1], brake=0))

        if action[0] != -0.75:
            print('Throttle:{}, Steer:{}'.format(0.5+0.5*action[0], action[1]))

        # This is very important since the car needs time to react with the apply_control() command.
        time.sleep(0.3)

        l_ = self.vehicle.get_location()
        distance_from_road = self.get_distance()
        alpha = self.get_tangent_angle_closest_point()
        beta, vcar = self.get_velocity_data()
        v_kmh = int(3.6*vcar)
        diff_v = v_kmh - last_velocity
        side = self.get_side()

        # REWARD FUNCTION
        # Path tracking reward
        if beta*alpha < 0:
            if abs(beta)+abs(alpha) > math.pi:
                path_tracking_reward = math.cos(2*math.pi-abs(beta) -
                                                abs(alpha))-0.1*distance_from_road
                if beta > alpha:
                    diff_angle = abs(beta)+abs(alpha)-2*math.pi
                else:
                    diff_angle = 2*math.pi-abs(beta)-abs(alpha)
            elif abs(beta)+abs(alpha) < math.pi:
                path_tracking_reward = math.cos(
                    abs(beta)+abs(alpha))-0.1*distance_from_road
                if beta > alpha:
                    diff_angle = abs(beta)+abs(alpha)
                else:
                    diff_angle = -abs(beta)-abs(alpha)
            else:
                path_tracking_reward = -1-0.1*distance_from_road
                diff_angle = math.pi
        else:
            path_tracking_reward = math.cos(
                abs(beta-alpha))-0.1*distance_from_road
            diff_angle = beta-alpha

        # Velocity tracking reward
        if diff_v <= 0 and v_kmh < 30:
            velocity_tracking_reward = (0.05/3)*(v_kmh-30)
        elif diff_v > 0 and v_kmh < 30:
            velocity_tracking_reward = 0
        elif v_kmh < 40:
            velocity_tracking_reward = 0.05*(v_kmh-30)
        elif v_kmh >= 40:
            velocity_tracking_reward = 0.5

        if v_kmh >= 45:  # Speed limit
            velocity_tracking_reward -= 0.2*(v_kmh-45)

        # Need to balance between these two rewards !!!
        reward = path_tracking_reward + velocity_tracking_reward
        reward = max(reward, -2.5)
        #############################################################################

        # For adjusting acceptable distance from road when training and testing
        if self.test == True:
            DISTANCE_FROM_ROAD = 5.0
            DISTANCE_TO_FINISH = 2.0
        else:
            DISTANCE_FROM_ROAD = 2.0
            DISTANCE_TO_FINISH = 1.0

        # Finalizing the reward of the episode
        if len(self.collision_hist) != 0 or distance_from_road >= DISTANCE_FROM_ROAD:
            done = True
            reward = -4
        else:
            done = False
            reward = reward

        if self.episode_start + settings.SECONDS_PER_EPISODE < time.time():
            done = True
            reward = reward

        if self.get_distance_to_finish() <= DISTANCE_TO_FINISH:
            done = True
            is_finished = True
            reward = reward
        if abs(action[1]-last_action[1]) >= 0.3:    # 0.4 -> 0.3
            reward -= 0.5

        # Return current state
        xx = int(self.distance_to_obstacle_f)
        yy = int(self.distance_to_obstacle_r)
        zz = int(self.distance_to_obstacle_l)

        img = np.random.rand(settings.IM_HEIGHT, settings.IM_WIDTH, 2)
        img[:, :, 0:1] = self.segm_img
        img[:, :, 1:2] = self.depth

        self.current_state = np.array(
            [img, xx, yy, zz, diff_angle, distance_from_road, side, last_action[1], diff_v, v_kmh], dtype='object')

        return self.current_state, reward, done, is_finished, None
