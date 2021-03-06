#!/usr/bin/env python

# Copyright (c) 2019 Aptiv
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
An example of client-side bounding boxes with basic car controls.

Controls:

    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys
import grpc
from messages import pub_sub_pb2_grpc, pub_sub_pb2
import config


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla
import cv2
import weakref
import random
import time
import queue
import math
from carla import ColorConverter as cc
import matplotlib.pyplot as plt
from module.g29control_obj import World, HUD, DualControl, CollisionSensor

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


def to_byte_array(npc_xyhs, ego_xyh, image):
    nbrs = []
    x = ego_xyh[0]
    y = ego_xyh[1]
    heading = ego_xyh[2]
    ego_id = ego_xyh[3]

    ego_data = [ego_id, 0., 0.]
    nbrs.append(ego_data)
    for npc_xyh in npc_xyhs:
        x = npc_xyh[0]
        y = npc_xyh[1]
        heading = npc_xyh[2]
        nbr_id = npc_xyh[3]
        nbr_data = [nbr_id, x, y]
        nbrs.append(nbr_data)

    nbrs = np.array(nbrs).tobytes()
    ego = np.array([ego_xyh[0], ego_xyh[1], heading]).tobytes()
    image_bytes = image.tobytes()
    n_veh = len(npc_xyhs)+1  # +1 ego
    return ego, nbrs, n_veh, image_bytes


ServerSetting = True
if ServerSetting:
    random.seed(30)
    start_time = time.time()
    use_dw = False
    dw_speed = 0.1
    global_percentage_speed_difference = -150
    global_distance_to_leading_vehicle = 1

    ##
    VIEW_WIDTH = 400
    VIEW_HEIGHT = 600
    VIEW_FOV = 90

    BB_COLOR = (248, 64, 24)
    tm_port = 2000

    #camloc, camrot = carla.Location(x=10, z=11.4*4), carla.Rotation(-90,0,0)
    camloc, camrot = carla.Location(x=0, z=50), carla.Rotation(-90, 0, 0)
    #camtype = 'sensor.camera.semantic_segmentation'
    camtype = 'sensor.camera.rgb'

    use_lpc = False
    lidar_channel = 32
    lidar_dist = 180
    lidar_pps = 1920*1080
    lidar_rot = 60

    use_bbox = False
    use_XYH = True

    num_npc = 110

    show_top_view = False
    pygamefps = 30
    fixed_delta_seconds = 1/pygamefps

    save_path = f'C:/CARLA_0.9.10/WindowsNoEditor/PythonAPI/examples/Saved/top_view/record_{len(os.listdir("C:/CARLA_0.9.10/WindowsNoEditor/PythonAPI/examples/Saved/top_view/"))+1}/'
    os.makedirs(save_path)

if True:
    load_world = "Town06"
else:
    load_world = None

SetAutoPilot = carla.command.SetAutopilot
FutureActor = carla.command.FutureActor
do_control_wheel = True
do_control_keyboard = False

GREEN = (0, 255, 0)

timeout = 15


# ==============================================================================
# -- ClientSideBoundingBoxes ---------------------------------------------------
# ==============================================================================

class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def get_XYHs(carla_client, ego_car):
        selfCar = ego_car
        ego_xyh = selfCar.get_transform()
        ego_id = selfCar.id
        #npc_list = [carla_client.world.get_actor(npc_id) for npc_id in carla_client.npc_list]
        npc_list = [car for car in list(carla_client.world.get_actors().filter("vehicle.*")) if car.id != ego_car.id]
        npc_xyhs = [ClientSideBoundingBoxes.get_XYH(npc, ego_xyh) for npc in npc_list]
        # return npc_xyhs, (ego_xyh.location.x, ego_xyh.location.y, ego_xyh.rotation.yaw/180*np.pi)
        return npc_xyhs, (ego_xyh.location.y, ego_xyh.location.x, ego_xyh.rotation.yaw/180*np.pi, ego_id)

    @staticmethod
    def get_XYH(npc, ego_xyh):
        ego_x = ego_xyh.location.x
        ego_y = ego_xyh.location.y
        ego_h = ego_xyh.rotation.yaw #degrees

        npc_xyh = npc.get_transform()
        npc_x = npc_xyh.location.x - ego_x
        npc_y = npc_xyh.location.y - ego_y
        npc_h = npc_xyh.rotation.yaw - ego_h #degrees

        npc_hr = npc_h/180 * np.pi
        ego_hr = ego_h/180*np.pi
        npc_yr = npc_x*np.cos(ego_hr) + npc_y*np.sin(ego_hr)
        npc_xr = npc_y*np.cos(ego_hr) - npc_x*np.sin(ego_hr)

        return npc_yr, npc_xr, npc_hr, npc.id
        # return npc_yr, npc_xr, npc_hr

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))

        vcount = 0;
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # draw lines
            # base
            # pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            # pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            # pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            # pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            # pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            if (0 <= points[4][0] and points[4][0] <= VIEW_WIDTH and 0 <=points[4][1] and points[4][1] <= VIEW_HEIGHT):
                vcount += 1
            # base-top
            # pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            # pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            # pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            # pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))
        #print(vcount, "/", len(bounding_boxes), end='\r')



    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """

        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    def _sensor_to_world(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """
        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform())
        world_cords = np.dot(sensor_world_matrix, cords)
        return world_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """

        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================


class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):
        self.client = None
        self.world = None
        self.camera = None
        self.car = None
        self.npc_list = []
        self.spawn_points = []
        self.image = None
        self.capture = True
        self.number_of_cars = 0
        self.collision_ego_id = None
        self.collision_other_id = None
        self.collision_time = None
        self.collision_sensors = []
        self.collision_attached = []

    def camera_blueprint(self):
        """
        Returns camera blueprint.
        """

        camera_bp = self.world.get_blueprint_library().find(camtype)
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        return camera_bp

    def lidar_blueprint(self):
        lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('channels',str(lidar_channel))
        lidar_bp.set_attribute('points_per_second',str(lidar_pps))
        lidar_bp.set_attribute('rotation_frequency',str(lidar_rot))
        lidar_bp.set_attribute('range',str(lidar_dist))
        return lidar_bp

    def collision_blueprint(self):
        collision_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        return collision_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        traffic_manager = self.client.get_trafficmanager(tm_port)

        traffic_manager.set_global_distance_to_leading_vehicle(global_distance_to_leading_vehicle)
        traffic_manager.global_percentage_speed_difference(global_percentage_speed_difference)
        traffic_manager.set_synchronous_mode(synchronous_mode)

        if synchronous_mode:
            settings.fixed_delta_seconds = fixed_delta_seconds
        else:
            settings.fixed_delta_seconds = None
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """
        # spawn_points = self.world.get_map().get_spawn_points()

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        print("car_bp : ", self.world.get_blueprint_library().filter('vehicle.*'))
        try:
            location = random.choice(self.spawn_points)
            self.car = self.world.spawn_actor(car_bp, location)
        except:
            print("collision in spawn point")
            self.setup_car()

    def setup_npc(self, number_of_npc=num_npc):
        self.spawn_points = self.world.get_map().get_spawn_points()
        number_of_spawn_points = len(self.spawn_points)

        if number_of_npc < number_of_spawn_points:
            random.shuffle(self.spawn_points)
        elif number_of_npc > number_of_spawn_points:
            number_of_npc = number_of_spawn_points

        blueprints = self.world.get_blueprint_library().filter('vehicle.*')
        batch = []
        for n, transform in enumerate(self.spawn_points):
            if n >= number_of_npc:
                break
            blueprint = random.choice(blueprints)
            #batch.append(carla.command.SpawnActor(blueprint, transform))
            blueprint.set_attribute('role_name', 'autopilot')
            batch.append(carla.command.SpawnActor(blueprint, transform).then(SetAutoPilot(FutureActor, True)))
            self.spawn_points.pop(0)

        for response in self.client.apply_batch_sync(batch):
            self.npc_list.append(response.actor_id)

    def setup_camera(self, attached_vehicle):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        camera_transform = carla.Transform(camloc, camrot)
        self.camera = self.world.spawn_actor(self.camera_blueprint(), camera_transform, attach_to=attached_vehicle)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration

    def setup_lidar(self):
        lidar_location = carla.Location(0,0,2)
        lidar_rotation = carla.Rotation(0,0,0)
        lidar_transform = carla.Transform(lidar_location,lidar_rotation)
        self.lidar_sen = self.world.spawn_actor(self.lidar_blueprint(), lidar_transform,attach_to=ego_vehicle,attachment_type=carla.AttachmentType.Rigid)
        self.lidar_sen.listen(lambda point_cloud: point_cloud.save_to_disk(save_path+'new_lidar_output/%.6d.ply' % point_cloud.frame))

    def setup_collision(self, attached_vehicle):
        self.collision_sensors.append(self.world.spawn_actor(self.collision_blueprint(), carla.Transform(), attach_to=attached_vehicle))
        self_id = attached_vehicle.id
        self.collision_sensors[-1].listen(lambda event: self.get_collision_id(self_id, event))

    def get_collision_id(self, ego_id, event):

        self.collision_ego_id = ego_id
        self.collision_other_id = event.other_actor.id
        self.collision_time = event.frame

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """

        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture:
            self.image = img
            if camtype == 'sensor.camera.semantic_segmentation':
                self.image.convert(cc.CityScapesPalette)
            #self.image.save_to_disk(save_path + 'new_sem_output/%.6d.jpg' % img.frame,)
            if show_top_view:
                i = np.array(self.image.raw_data)
                i = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
                i = i[:, :, :3]

                cv2.imshow("top_view", i)
                cv2.waitKey(1)
                # plt.imshow(i)
                # plt.show()
            self.capture = False

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """
        #return 0
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = array.swapaxes(0, 1)
            surface = pygame.surfarray.make_surface(array)
            display.blit(surface, (0, 0))

    def get_image(self):
        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            array = array.swapaxes(0, 1)
        else:
            array = np.zeros((VIEW_WIDTH, VIEW_HEIGHT, 3))

        return array

    def game_loop(self, stub):
        """
        Main program loop.
        """
        filter = "vehicle.*"

        try:
            pygame.init()

            self.client = carla.Client('203.246.114.177', tm_port)
            self.client.set_timeout(timeout)
            self.world = self.client.get_world()

            if load_world != None:
                print(self.client.get_available_maps())
                self.client.load_world(load_world)
                self.client.reload_world()

                time.sleep(3)

            self.setup_npc()

            if use_lpc:
                self.setup_lidar()
            # hud = HUD(1920, 1080)
            hud = HUD(1280, 720)
            world = World(self.world, hud, filter)
            if do_control_wheel:
                display = pygame.display.set_mode((1280, 720), pygame.HWSURFACE | pygame.DOUBLEBUF)
                # display = pygame.display.set_mode((1920, 1080), pygame.HWSURFACE | pygame.DOUBLEBUF)


                controller = DualControl(world, None)
                self.setup_camera(world.player)
                self.setup_collision(world.player)
            elif do_control_keyboard:
                self.setup_car()
                self.setup_camera(self.car)
                display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)

            pygame.display.set_caption("Ryu Jeong Hwan")
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)
            npc_list = [self.world.get_actor(npc_id) for npc_id in self.npc_list]

            counter = 0

            while True:
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(pygamefps)
                if counter % 100 == 99:
                    vehicles = self.world.get_actors().filter('vehicle.*')
                    for _v in vehicles:
                        if _v not in npc_list and _v.id not in self.collision_attached and _v.id != world.player.id:
                            print(f"another player is {_v.id}")
                            self.collision_attached.append(_v.id)
                            self.setup_collision(_v)

                if True:
                    for _v in npc_list:
                        target_velocity = _v.get_velocity()
                        _speed = np.sqrt(target_velocity.x ** 2 + target_velocity.y ** 2 + target_velocity.z ** 2) + 0.01
                        if _speed > 150/3.6:
                            target_velocity *= 0.8
                            target_velocity.z = -0.5
                            _v.set_target_velocity(target_velocity)
                            continue
                        if _speed > 80 / 3.6:
                            continue
                        if np.random.uniform() < 0.93:
                            if np.random.uniform() < 0.01:
                                target_velocity *= 0.4
                                _v.set_target_velocity(target_velocity)
                            continue

                        #target_velocity *= 70 / _speed / 3.6
                        target_velocity *= 1.1
                        target_velocity.z = -0.5
                        _v.set_target_velocity(target_velocity)

                if do_control_wheel:
                    if controller.parse_events(world, pygame_clock):
                        return
                    world.tick(pygame_clock)
                    world.render(display)
                elif do_control_keyboard:
                    if self.control(self.car):
                        return
                    self.render(display)
                image = self.get_image()

                if use_bbox:
                    bounding_boxes = ClientSideBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                    ClientSideBoundingBoxes.draw_bounding_boxes(display, bounding_boxes)

                if use_XYH:
                    cam_info = self.camera.get_transform()
                    cam_loc = (cam_info.location.y, cam_info.location.x)
                    if do_control_wheel:
                        npc_xyhs, ego_xyh = ClientSideBoundingBoxes.get_XYHs(self, world.player)
                    elif do_control_keyboard:
                        npc_xyhs, ego_xyh = ClientSideBoundingBoxes.get_XYHs(self, self.car)
                    # print("\r", npc_xyhs, ego_xyh, int(counter*fixed_delta_seconds*10), end='')
                    print("\r", ego_xyh, int(counter*fixed_delta_seconds*10), end='')
                    ego, nbr, n_veh, image_bytes = to_byte_array(npc_xyhs, ego_xyh, image)
                    if self.collision_other_id is not None:
                        stub.UpdateHistory(pub_sub_pb2.RawTopic(frame=int(counter*fixed_delta_seconds*10),
                                                                image=image_bytes,
                                                                collision_other_id=self.collision_other_id,
                                                                ego_id=ego_xyh[3],
                                                                ego=ego, nbr=nbr, n_veh=n_veh))
                        self.collision_other_id = None
                        self.collision_ego_id = None
                    else:
                        stub.UpdateHistory(pub_sub_pb2.RawTopic(frame=int(counter * fixed_delta_seconds * 10),
                                                                image=image_bytes,
                                                                collision_other_id=-1,
                                                                ego_id=ego_xyh[3],
                                                                ego=ego, nbr=nbr, n_veh=n_veh))
                counter += 1

                pygame.display.flip()

                pygame.event.pump()

                #print(self.car.get_location(), end='\r')

        finally:
            print("\n destroy")
            self.set_synchronous_mode(False)
            self.camera.destroy()
            if use_lpc:
                self.lidar.destroy()
            # self.car.destroy()
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_list])
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.world.get_actors().filter('vehicle.*')])

            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    MAX_MESSAGE_LENGTH = 730000
    """
    Initializes the client-side bounding box demo.
    """
    with grpc.insecure_channel("localhost:50054",
                               options=[
                                   ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                                   ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH)
                               ]) as channel:
        stub = pub_sub_pb2_grpc.RawDataStub(channel)

        try:
            client = BasicSynchronousClient()
            client.game_loop(stub)
        finally:
            print('EXIT')


if __name__ == '__main__':
    main()
