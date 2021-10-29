import numpy as np
from collections import deque

import config


class DataSet:
    def __init__(self):
        self.ego_nbr_q = deque([], maxlen=1000)
        self.t_q = deque([], maxlen=1000)

    def get_data(self, frame):
        for i, collected_t in enumerate(self.t_q):
            if frame == collected_t:
                data = self.ego_nbr_q[i]
                return data
        return -1

    def update_data(self, frame, data):
        cnt = self.t_q.count(frame)
        if cnt == 0:
            self.t_q.append(frame)
            self.ego_nbr_q.append(data)
        else:
            # print("not")
            pass

    def check_data(self, frame):
        cnt = self.t_q.count(frame)
        if cnt == 1:
            flag = 1
        else:
            if self.t_q[0] > frame:
                flag = 0  # not yet start
            elif self.t_q[-1] < frame:
                flag = 2  # wait for
            else:
                flag = 3  # something error

        return flag


class Vehicles:
    def __init__(self, ego_id, history_size, downsampling_rate, scene_size, grid_size):
        self.ego_id = ego_id
        self.vehicles = {}
        self.global_x = 0.0
        self.global_y = 0.0
        self.history_size = history_size
        self.downsampling_rate = downsampling_rate
        self.scene_size = scene_size
        self.grid_size = grid_size
        self.frame = 0.

    def stack_data(self, ego_data, nbr_data):
        self.global_x = ego_data[0]
        self.global_y = ego_data[1]

        nbr_vehicles = []

        for info in nbr_data:
            vehicle_id = info[0]
            nbr_vehicles.append(vehicle_id)

            # relative coordinates
            rc_x = info[1]
            rc_y = info[2]

            # global coordinates
            gc_x = self.global_x + rc_x
            gc_y = self.global_y + rc_y

            if vehicle_id in self.vehicles.keys():
                self.vehicles[vehicle_id].append((gc_x, gc_y))
            else:
                self.vehicles[vehicle_id] = deque([(gc_x, gc_y)] * 30, maxlen=30)

        # if stacked vehicle is not observed, the vehicle will be deleted
        del_list = []
        for stacked_v in self.vehicles.keys():
            if not stacked_v in nbr_vehicles:
                if not stacked_v is self.ego_id:
                    del_list.append(stacked_v)

        for del_id in del_list:
            del (self.vehicles[del_id])

    def get_grid_pos(self, x, y):
        grid_height = self.scene_size[0] / self.grid_size[0]
        grid_width = self.scene_size[1] / self.grid_size[1]

        # get the row
        for i in range((self.grid_size[0] // 2) + 1):
            if abs(x) <= (grid_height * i) + (grid_height / 2):
                r = (self.grid_size[0] // 2) - i
                break

        if x < 0:
            r = (self.grid_size[0] - 1) + (r * (-1))

        # get the col
        if abs(y) <= (grid_width / 2):
            c = 1
        else:
            if y > 0:
                c = 2
            else:
                c = 0

        return (r, c)

    def search_veh(self, id):
        if len(self.vehicles[id]) >= self.history_size:
            all_nbr_vehicles = list(self.vehicles.keys())
            all_nbr_vehicles.remove(id)

            origin_x = self.vehicles[id][-1][0]
            origin_y = self.vehicles[id][-1][1]

            nbr_vehicles = []
            grid_pos = []
            for nbr_vehicle in all_nbr_vehicles:
                cur_pos_x = self.vehicles[nbr_vehicle][-1][0] - origin_x
                cur_pos_y = self.vehicles[nbr_vehicle][-1][1] - origin_y

                if abs(cur_pos_x) <= self.scene_size[0] / 2 and abs(cur_pos_y) <= self.scene_size[1] / 2:
                    nbr_vehicles.append(nbr_vehicle)

                    grid_pos.append(self.get_grid_pos(cur_pos_x, cur_pos_y))

            if len(grid_pos) != len(set(grid_pos)) or (6, 0) in grid_pos:
                # print("Some vehicles are in same grid. not update")
                return -1, -1

            else:
                return nbr_vehicles, grid_pos

        else:
            print(id, "vehicle has insufficient historical trajectory. in", self.frame)
            return -1, -1

    def get_scene(self, id, nbr_vehicles, grid_pos):
        # get grid for visualizing
        scene_vehicles = np.zeros((config.GRID_SIZE[0], config.GRID_SIZE[1]))
        for idx, nbr_id in enumerate(nbr_vehicles):
            scene_vehicles[grid_pos[idx][0], grid_pos[idx][1]] = nbr_vehicles[idx]

        # get input fake_data
        scene_trajs = np.zeros((config.GRID_SIZE[0], config.GRID_SIZE[1], int(self.history_size) // self.downsampling_rate + 1, 2))
        traj_sample_idx = list(range(0, int(self.history_size) // self.downsampling_rate + 1, 1))
        traj_sample_idx.reverse()

        used_veh = []

        x_origin = np.array(self.vehicles[id])[traj_sample_idx[0]][0]
        y_origin = np.array(self.vehicles[id])[traj_sample_idx[0]][1]

        for idx, nbr_id in enumerate(nbr_vehicles):
            traj = np.array(self.vehicles[nbr_id])[traj_sample_idx]
            traj[:, 0] -= x_origin
            traj[:, 1] -= y_origin

            scene_trajs[grid_pos[idx][0], grid_pos[idx][1]] = traj
            used_veh.append(nbr_id)

        traj = np.array(self.vehicles[id])[traj_sample_idx]
        traj[:, 0] -= x_origin
        traj[:, 1] -= y_origin
        scene_trajs[config.EGO_POS[0], config.EGO_POS[1]] = traj

        return scene_vehicles, scene_trajs, used_veh

    def get_inputs(self, id, frame, show_grid=False):
        self.frame = frame
        inputs = {}
        used_list = []

        nbr_vehicles, grid_pos = self.search_veh(id)

        if not nbr_vehicles == -1:
            for i, nbr_id in enumerate(nbr_vehicles):

                nbr_nbr_vehicles, nbr_grid_pos = self.search_veh(nbr_id)
                if not nbr_nbr_vehicles == -1:
                    nbr_scene_vehicles, nbr_scene_traj, used_veh = self.get_scene(nbr_id, nbr_nbr_vehicles,
                                                                                  nbr_grid_pos)

                    inputs[nbr_id] = nbr_scene_traj
                    used_list += used_veh
                else:
                    return -1, -1
            scene_vehicles, scene_traj, used_veh = self.get_scene(id, nbr_vehicles, grid_pos)
            inputs[id] = scene_traj
            used_list += used_veh

            used_list_unique = list(set(used_list))
            for u_id in used_list_unique:
                del self.vehicles[u_id][0]

            scene_vehicles[config.EGO_POS[0], config.EGO_POS[1]] = id

            if show_grid:
                print(scene_vehicles)
            return inputs, scene_vehicles.transpose()

        else:
            return -1, -1
