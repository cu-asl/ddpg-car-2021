import sys
import os
import glob

import carla
import math
import time
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
    #BRAKE_AMT = 1.0
    actor_list = []
    collision_hist = []
    # pt_cloud = []
    # pt_cloud_filtered = []

    def __init__(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(180.0)

        # Load layered map for Town03 with minimum layout plus buildings and parked vehicles
        # self.world = self.client.get_world()
        self.world = self.client.load_world(
            'Town03_Opt', carla.MapLayer.Buildings | carla.MapLayer.ParkedVehicles)

        # Toggle all buildings off
        # self.world.unload_map_layer(carla.MapLayer.Buildings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter('model3')[0]
        # self.truck_2 = self.blueprint_library.filter('carlamotors')[0]

        spectator = self.world.get_spectator()
        spectator.set_transform(carla.Transform(
            carla.Location(56.2, -4.2, 3), carla.Rotation(yaw=180)))

        for x in list(self.world.get_actors()):
            if 'vehicle' in x.type_id or 'sensor' in x.type_id:
                x.destroy()

        self.blueprint_library = self.world.get_blueprint_library()
        self.blueprint = self.blueprint_library.filter('model3')[0]
        # self.blueprint = blueprint_library.filter('blueprint')[0]

        refmap = pd.read_csv('refmap3.csv')
        self.kd_tree_map = kdtree(refmap.values)

    def reset(self):
        self.collision_hist = []
        self.actor_list = []
        # self.pt_cloud = []
        # self.pt_cloud_filtered = []

        # Vehicle
        transform = carla.Transform(carla.Location(
            56.2, -4.2, 0.5), carla.Rotation(0, 180, 0))
        self.vehicle = self.world.spawn_actor(self.blueprint, transform)
        self.actor_list.append(self.vehicle)

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
        # sleep to get things started and to not detect a collision when the car spawns/falls from sky.
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
        self.vehicle.apply_control(
            carla.VehicleControl(throttle=0.0, brake=0.0))

        # Current state
        alpha = self.get_tangent_angle_closest_point()
        beta = alpha
        diff_angle = beta-alpha

        distance_from_road = self.get_distance()

        side = 1    # side = self.get_side()
        last_action = [0, 0]  # last_action [throttle, steer]
        diff_v = 0
        v_kmh = 0

        # distance_f = int(self.distance_to_obstacle_f)
        # distance_l = int(self.distance_to_obstacle_l)
        # distance_r = int(self.distance_to_obstacle_r)

        xx = self.distance_to_obstacle_f
        yy = self.distance_to_obstacle_r
        zz = self.distance_to_obstacle_l
        if xx <= 3:
            xx = 2
        elif xx <= 1:
            xx = 1
        else:
            xx = 0
        if yy <= 3:
            yy = 2
        elif yy <= 1:
            yy = 1
        else:
            yy = 0
        if zz <= 3:
            zz = 2
        elif zz <= 1:
            zz = 1
        else:
            zz = 0

        self.current_state = np.array(
            [xx, yy, zz, diff_angle, distance_from_road, side, last_action[1], diff_v, v_kmh])

        return self.current_state

    # def Black_screen(self):
    #     settings = self.world.get_settings()
    #     settings.no_rendering_mode = True
    #     self.world.apply_settings(settings)

    # def set_location(self,x,y):

    #     self.lo_x,self.lo_y=x,y
    #     self.place=x,y
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
    #############################################################################

    def collision_data(self, event):
        self.collision_hist.append(event)

    #############################################################################
    def process_lidar(self, raw):
        points = np.frombuffer(raw.raw_data, dtype=np.dtype('f4'))
        # points = np.reshape(points, (int(points.shape[0] / 3), 3))*np.array([1,-1,-1])
        points = np.reshape(
            points, (int(points.shape[0] / 4), 4))[:, :3]*np.array([1, -1, -1])

        lidar_f = self.lidar_line(points, 90, 2)
        lidar_r = self.lidar_line(points, 45, 2)
        lidar_l = self.lidar_line(points, 135, 2)

        if len(lidar_f) == 0:
            pass
        else:
            self.distance_to_obstacle_f = min(lidar_f[:, 1])-2.482

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
    ##############################################################################################

    def step(self, action, last_action, last_velocity):

        action[0] = abs(action[0])
        action = [act.item() for act in action]

        self.vehicle.apply_control(carla.VehicleControl(
            throttle=action[0], steer=action[1], brake=0))

        time.sleep(0.3)

        l_ = self.vehicle.get_location()
        distance_from_road = self.get_distance()
        alpha = self.get_tangent_angle_closest_point()
        beta, vcar = self.get_velocity_data()
        v_kmh = int(3.6*vcar)
        diff_v = v_kmh - last_velocity
        side = self.get_side()

        # distance_f = int(self.distance_to_obstacle_f)
        # distance_l = int(self.distance_to_obstacle_l)
        # distance_r = int(self.distance_to_obstacle_r)

        if beta*alpha < 0:
            if abs(beta)+abs(alpha) > math.pi:
                reward = math.cos(2*math.pi-abs(beta) -
                                  abs(alpha))-0.1*distance_from_road
                if beta > alpha:
                    diff_angle = abs(beta)+abs(alpha)-2*math.pi
                else:
                    diff_angle = 2*math.pi-abs(beta)-abs(alpha)
            elif abs(beta)+abs(alpha) < math.pi:
                reward = math.cos(abs(beta)+abs(alpha))-0.1*distance_from_road
                if beta > alpha:
                    diff_angle = abs(beta)+abs(alpha)
                else:
                    diff_angle = -abs(beta)-abs(alpha)
            else:
                reward = -1-0.1*distance_from_road
                diff_angle = math.pi
        else:
            reward = math.cos(abs(beta-alpha))-0.1*distance_from_road
            diff_angle = beta-alpha

        # NEW ###############################
        if diff_v <= 0 and v_kmh < 30:
            reward -= (0.05/3)*(30-v_kmh)
        elif diff_v > 0 and v_kmh < 30:
            reward = reward
        elif v_kmh < 50:
            reward += 0.01*(v_kmh-30)
        elif v_kmh > 50:
            reward += 0.2
        else:
            print("Reward is not calculated.")
        #############################################################################

        if len(self.collision_hist) != 0 or distance_from_road >= 5.0:
            done = True
            reward = -3  # -2
        else:
            done = False
            reward = reward

        if self.episode_start + settings.SECONDS_PER_EPISODE < time.time():
            done = True
            reward = reward
        if l_.y <= -55:
            done = True
            reward = reward
        if abs(action[1]-last_action[1]) >= 0.4:
            reward -= 0.5

        xx = self.distance_to_obstacle_f
        yy = self.distance_to_obstacle_r
        zz = self.distance_to_obstacle_l
        if xx <= 3:
            xx = 2
        elif xx <= 1:
            xx = 1
        else:
            xx = 0
        if yy <= 3:
            yy = 2
        elif yy <= 1:
            yy = 1
        else:
            yy = 0
        if zz <= 3:
            zz = 2
        elif zz <= 1:
            zz = 1
        else:
            zz = 0

        self.current_state = np.array(
            [xx, yy, zz, diff_angle, distance_from_road, side, last_action[1], diff_v, v_kmh])

        return self.current_state, reward, done, None

    # def test_step(self, action, last_action):

    #     finish = 0

    #     strr = action

    #     self.vehicle.apply_control(carla.VehicleControl(
    #         throttle=0.3, brake=0, steer=strr))
    #     time.sleep(0.5)

    #     l_ = self.vehicle.get_location()
    #     ref_map = self.get_distance()
    #     alpha = self.get_tangent_angle_closest_point()
    #     beta, vcar = self.get_velocity_data()
    #     side = self.get_side()

    #     if beta*alpha < 0:
    #         if abs(beta)+abs(alpha) > math.pi:
    #             reward = math.cos(2*math.pi-abs(beta)-abs(alpha))-0.1*ref_map
    #             if beta > alpha:
    #                 diff_angle = abs(beta)+abs(alpha)-2*math.pi
    #             else:
    #                 diff_angle = 2*math.pi-abs(beta)-abs(alpha)
    #         elif abs(beta)+abs(alpha) < math.pi:
    #             reward = math.cos(abs(beta)+abs(alpha))-0.1*ref_map
    #             if beta > alpha:
    #                 diff_angle = abs(beta)+abs(alpha)
    #             else:
    #                 diff_angle = -abs(beta)-abs(alpha)
    #         else:
    #             reward = -1-0.1*ref_map
    #             diff_angle = math.pi
    #     else:
    #         reward = math.cos(abs(beta-alpha))-0.1*ref_map
    #         diff_angle = beta-alpha

    #     if len(self.collision_hist) != 0:
    #         done = True
    #         reward = -2
    #     else:
    #         done = False
    #         reward = reward

    #     if self.episode_start + settings.SECONDS_PER_EPISODE < time.time():
    #         done = True
    #         reward = reward
    #     if l_.x <= 145:
    #         done = True
    #         finish = 1
    #         reward = reward
    #     if abs(action-last_action) >= 0.4:
    #         reward -= 0.5

    #     xx = self.distance_to_obstacle_f
    #     yy = self.distance_to_obstacle_r
    #     zz = self.distance_to_obstacle_l
    #     if xx <= 3:
    #         xx = 2
    #     elif xx <= 1:
    #         xx = 1
    #     else:
    #         xx = 0
    #     if yy <= 3:
    #         yy = 2
    #     elif yy <= 1:
    #         yy = 1
    #     else:
    #         yy = 0
    #     if zz <= 3:
    #         zz = 2
    #     elif zz <= 1:
    #         zz = 1
    #     else:
    #         zz = 0
    #     state_ = np.array([xx, yy, zz, diff_angle, ref_map, side, last_action])

    #     return state_, reward, done, finish, None
