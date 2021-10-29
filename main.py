from messages import pub_sub_pb2_grpc, pub_sub_pb2
from concurrent import futures
import grpc
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from time import gmtime, strftime

from module.utils import DataSet
import config


raw_dataset = DataSet()
grid_dataset = DataSet()
pred_dataset = DataSet()
risk_dataset = DataSet()

dir_name = strftime("%m-%d-%H-%M", gmtime())
if not os.path.isdir(os.path.join("log", dir_name)):
    os.mkdir(os.path.join("log", dir_name))

class RawDataServicer(pub_sub_pb2_grpc.RawDataServicer):
    def __init__(self):
        pass

    def UpdateHistory(self, raw_topic, context):
        frame = raw_topic.frame
        image = np.frombuffer(raw_topic.image, dtype=np.uint8).reshape(400, 600, 3)
        ego = np.frombuffer(raw_topic.ego)
        nbr = np.frombuffer(raw_topic.nbr).reshape(raw_topic.n_veh, 3)
        collision_other_id = raw_topic.collision_other_id
        ego_id = raw_topic.ego_id
        #if collision_other_id > 0:
        if True:
            print("\r!!!!!!!!!!!!!!!!!!!!!!!!!!!!!collision!!!!", collision_other_id, end='')
        data = [image, ego, nbr, raw_topic.n_veh, collision_other_id, ego_id]
        # print("\r", frame, end="")
        raw_dataset.update_data(frame, data)

        raw_data_dict = {"image": data[0], "ego": data[1], "nbr": data[2], "n_veh": data[3], "collision_other_id": data[4],
         "ego_id": data[5]}

        with open(f"log/{dir_name}/{frame}.pkl", "wb") as f:
            pickle.dump(raw_data_dict, f)

        return pub_sub_pb2.Flag(flag=1)


class MakeInputsServicer(pub_sub_pb2_grpc.MakeInputsServicer):
    def __init__(self):
        pass

    def CheckFrame(self, request, context):
        flag = raw_dataset.check_data(request.frame)
        return pub_sub_pb2.Flag(flag=flag)

    def GetHistory(self, request, context):
        data = raw_dataset.get_data(request.frame)
        if data == -1:
            return -1
        image, ego_info, nbr_info, n_veh, collision_other_id, collision_ego_id = data
        image = image.tobytes()
        ego_info = ego_info.tobytes()
        nbr_info = nbr_info.tobytes()
        return pub_sub_pb2.RawTopic(frame=request.frame, image=image, ego=ego_info, nbr=nbr_info, n_veh=n_veh)

    def UpdateGrid(self, grid_topic, context):
        frame = grid_topic.frame
        trajectory = np.frombuffer(grid_topic.trajectory)
        scene = np.frombuffer(grid_topic.scene)
        veh_ids = np.frombuffer(grid_topic.veh_ids, dtype=np.int32)
        n_veh = grid_topic.n_veh
        ego_id = grid_topic.ego_id

        trajectory = trajectory.reshape([n_veh,
                                         config.GRID_SIZE[0]*config.GRID_SIZE[1],
                                         int(config.HISTORY_SIZE) // config.DOWNSAMPLING_RATE + 1, 2])  # (x, y)
        scene = scene.reshape([config.GRID_SIZE[0], config.GRID_SIZE[1]])
        data = [ego_id, veh_ids, trajectory, scene]
        grid_dataset.update_data(frame, data)

        # print("UpdateGrid", frame)
        return pub_sub_pb2.Flag(flag=1)


class NetworkForwardServicer(pub_sub_pb2_grpc.NetworkForwardServicer):
    def __init__(self):
        pass

    def CheckFrame(self, request, context):
        flag = grid_dataset.check_data(request.frame)
        return pub_sub_pb2.Flag(flag=flag)

    def GetGrid(self, request, context):
        data = grid_dataset.get_data(request.frame)
        if data == -1:
            return -1
        ego_id, veh_ids, trajectory, scene = data
        n_veh = len(veh_ids)
        veh_ids = veh_ids.tobytes()
        trajectory = trajectory.tobytes()
        scene = scene.tobytes()
        return pub_sub_pb2.GridTopic(frame=request.frame,
                                     trajectory=trajectory,
                                     scene=scene,
                                     veh_ids=veh_ids,
                                     n_veh=n_veh,
                                     ego_id=ego_id)

    def UpdatePred(self, pred_topic, context):
        frame = pred_topic.frame
        n_veh = pred_topic.n_veh
        ego_i = pred_topic.ego_i
        pred_trajectory = np.frombuffer(pred_topic.pred_trajectory)
        veh_ids = np.frombuffer(pred_topic.veh_ids, dtype=np.int32)
        scene = np.frombuffer(pred_topic.scene)

        pred_trajectory = pred_trajectory.reshape([n_veh,
                                                   int(config.FUTURE_SIZE) // config.DOWNSAMPLING_RATE, 2])
        scene = scene.reshape([config.GRID_SIZE[0], config.GRID_SIZE[1]])

        data = [veh_ids, ego_i, pred_trajectory, scene]
        pred_dataset.update_data(frame, data)

        return pub_sub_pb2.Flag(flag=1)


class RiskAssessmentServicer(pub_sub_pb2_grpc.RiskAssessmentServicer):
    def __init__(self):
        pass

    def CheckFrame(self, request, context):
        flag = pred_dataset.check_data(request.frame)
        return pub_sub_pb2.Flag(flag=flag)

    def GetPred(self, request, context):
        data = pred_dataset.get_data(request.frame)
        if data == -1:
            return -1
        veh_ids, ego_i, pred_trajectory, scene = data

        return pub_sub_pb2.PredTopic(frame=request.frame,
                                     pred_trajectory=pred_trajectory.tobytes(),
                                     veh_ids=veh_ids.tobytes(),
                                     scene=scene.tobytes(),
                                     n_veh=len(veh_ids),
                                     ego_i=ego_i)

    def UpdateRisk(self, risk_topic, context):
        frame = risk_topic.frame
        risk_2darray = np.frombuffer(risk_topic.risk)
        n_collision = risk_topic.n_collision
        ego_id = risk_topic.ego_id

        risk_2darray = risk_2darray.reshape([n_collision, 3])
        # risks : [(collision_time, id, exp_ttc), ...]
        data = [risk_2darray, n_collision, ego_id]
        risk_dataset.update_data(frame, data)

        return pub_sub_pb2.Flag(flag=1)


class VisualizeServicer(pub_sub_pb2_grpc.VisualizeServicer):
    def __init__(self):
        pass

    def CheckFrame(self, request, context):
        # flag = raw_dataset.check_data(request.frame)
        flag = grid_dataset.check_data(request.frame)
        # flag = pred_dataset.check_data(request.frame)
        # flag = risk_dataset.check_data(request.frame)
        return pub_sub_pb2.Flag(flag=flag)

    def GetHistory(self, request, context):
        data = raw_dataset.get_data(request.frame)
        if data == -1:
            return -1
        image, ego_info, nbr_info, n_veh, collision_other_id, collision_ego_id = data
        image = image.tobytes()
        ego_info = ego_info.tobytes()
        nbr_info = nbr_info.tobytes()

        return pub_sub_pb2.RawTopic(frame=request.frame, image=image, ego=ego_info, nbr=nbr_info, n_veh=n_veh,
                                    collision_other_id=collision_other_id, ego_id=collision_ego_id)

    def GetGrid(self, request, context):
        data = grid_dataset.get_data(request.frame)
        if data == -1:
            return -1
        ego_id, veh_ids, trajectory, scene = data
        n_veh = len(veh_ids)
        veh_ids = veh_ids.tobytes()
        trajectory = trajectory.tobytes()
        scene = scene.tobytes()
        return pub_sub_pb2.GridTopic(frame=request.frame,
                                     trajectory=trajectory,
                                     scene=scene,
                                     veh_ids=veh_ids,
                                     n_veh=n_veh,
                                     ego_id=ego_id)

    def GetPred(self, request, context):
        data = pred_dataset.get_data(request.frame)
        if data == -1:
            return -1
        veh_ids, ego_i, pred_trajectory, scene = data

        return pub_sub_pb2.PredTopic(frame=request.frame,
                                     pred_trajectory=pred_trajectory.tobytes(),
                                     veh_ids=veh_ids.tobytes(),
                                     scene=scene.tobytes(),
                                     n_veh=len(veh_ids),
                                     ego_i=ego_i)

    def GetRisk(self, request, context):
        data = risk_dataset.get_data(request.frame)
        if data == -1:
            return -1
        risk_2darray, n_collision, ego_id = data

        return pub_sub_pb2.RiskTopic(frame=request.frame,
                                     risk=risk_2darray.tobytes(),
                                     n_collision=n_collision,
                                     ego_id=ego_id)


def serve():
    MAX_MESSAGE_LENGTH = 730000
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=[
                             ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                             ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH)
                            ]
                         )

    pub_sub_pb2_grpc.add_RawDataServicer_to_server(RawDataServicer(), server)
    pub_sub_pb2_grpc.add_MakeInputsServicer_to_server(MakeInputsServicer(), server)
    pub_sub_pb2_grpc.add_NetworkForwardServicer_to_server(NetworkForwardServicer(), server)
    pub_sub_pb2_grpc.add_RiskAssessmentServicer_to_server(RiskAssessmentServicer(), server)
    pub_sub_pb2_grpc.add_VisualizeServicer_to_server(VisualizeServicer(), server)

    server.add_insecure_port("localhost:50053")
    server.add_insecure_port("localhost:50054")
    server.add_insecure_port("localhost:50055")
    server.add_insecure_port("localhost:50056")

    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
