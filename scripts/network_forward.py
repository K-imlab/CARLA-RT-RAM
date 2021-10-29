import grpc
import numpy as np
import time

import config
from messages import pub_sub_pb2_grpc, pub_sub_pb2
from module import model

route_predictor = model.weight_load()


def from_bytes(topic):
    '''
    :param topic:
    trajectory : shape(n_nbrs,39,16,2)
    veh_ids : shape(n_vehs) ids of nbr vehicle
    scene : shape(39) ids of grids
    :return:
    '''
    trajectory = np.frombuffer(topic.trajectory)
    # (n_veh, 39, 16, 2)
    trajectory = trajectory.reshape([topic.n_veh,
                                     config.GRID_SIZE[0] * config.GRID_SIZE[1],
                                     int(config.HISTORY_SIZE) // config.DOWNSAMPLING_RATE + 1, 2])

    veh_ids = np.frombuffer(topic.veh_ids, dtype=np.int32)
    ego_i = np.where(veh_ids == topic.ego_id)[0][0]

    scene = np.frombuffer(topic.scene)
    return trajectory, veh_ids, scene, ego_i


def run():
    frame = 0
    with grpc.insecure_channel("localhost:50055") as channel:
        stub = pub_sub_pb2_grpc.NetworkForwardStub(channel)
        while True:
            st_time = time.time()
            flag_topic = stub.CheckFrame(pub_sub_pb2.TimeRequest(frame=frame))
            flag = flag_topic.flag
            if flag == 1:
                grid_topic = stub.GetGrid(pub_sub_pb2.TimeRequest(frame=frame))
            elif flag == 0:
                print("\r: There are no ego vehicles yet in", frame, end="")
                frame += config.DOWNSAMPLING_RATE
                time.sleep(0.1)
                continue
            elif flag == 2:
                print(f"\r: Waiting for network inputs {frame}", end="")
                time.sleep(0.1)
                continue
            else:
                print(f"\r: It was omitted from the make_input process. {frame}", end="")
                frame += config.DOWNSAMPLING_RATE
                time.sleep(0.1)
                continue

            trajectory, veh_ids, scene, ego_i = from_bytes(grid_topic)
            hists = trajectory[:, config.EGO_IDX, :, :]
            nbrss = trajectory.copy()
            nbrss[:, config.EGO_IDX, :, :] = 0
            preds = route_predictor.predict([hists, nbrss])
            preds = np.array(preds).astype(np.float64)
            preds = preds.tobytes()
            stub.UpdatePred(pub_sub_pb2.PredTopic(frame=frame,
                                                  pred_trajectory=preds,
                                                  veh_ids=veh_ids.tobytes(),
                                                  scene=scene.tobytes(),
                                                  n_veh=len(veh_ids),
                                                  ego_i=ego_i))

            print("elapsed time : ", time.time() - st_time, "UpdatePred", frame)
            frame += config.DOWNSAMPLING_RATE
            # time.sleep(0.1)


if __name__ == "__main__":
    run()
