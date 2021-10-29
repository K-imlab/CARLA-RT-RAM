import grpc
import numpy as np
import time

from module.utils import Vehicles
from messages import pub_sub_pb2_grpc, pub_sub_pb2
import config


np.set_printoptions(suppress=True)


def to_bytes(id_traj_dict, scene):
    keys = list(id_traj_dict.keys())
    number_veh = len(keys)
    shape = id_traj_dict[config.EGO_ID].shape
    idxs = list(range(0, shape[0]))
    idxs.reverse()
    inputs = []
    for veh_id in keys:
        input1 = np.zeros([shape[1], shape[0], shape[2], shape[3]])
        # left-down side is 0
        id_traj_dict[veh_id][:, :, :, :] = id_traj_dict[veh_id][idxs, :, :, :]
        for i in range(shape[1]):
            input1[i, :, :, :] = id_traj_dict[veh_id][:, i, :, :]
        inputs.append(input1)
    inputs = np.array(inputs)
    # print(np.round(inputs[0][1][6], 4))
    print(inputs[0, :,:, 0, 0])

    inputs = inputs.tobytes()  # (n_veh, 39, 16, 2)
    veh_ids = np.array(keys).tobytes()
    scene = np.array(scene).tobytes()

    return inputs, veh_ids, number_veh, scene


def run():
    frame = 0
    vehicles = Vehicles(config.EGO_ID,
                        config.HISTORY_SIZE,
                        config.DOWNSAMPLING_RATE,
                        config.SCENE_SIZE,
                        config.GRID_SIZE)
    with grpc.insecure_channel("localhost:50054") as channel:
        stub = pub_sub_pb2_grpc.MakeInputsStub(channel)
        while True:
            st_time = time.time()

            flag_topic = stub.CheckFrame(pub_sub_pb2.TimeRequest(frame=frame))
            flag = flag_topic.flag
            if flag == 1:
                raw_topic = stub.GetHistory(pub_sub_pb2.TimeRequest(frame=frame))
                print("GetHistory Call", frame)
            elif flag == 0:
                print("\r: There are no ego vehicles yet in", frame, end="")
                frame += config.DOWNSAMPLING_RATE
                time.sleep(0.1)
                continue
            else:
                print(f"\r: Waiting for simulator {flag} {frame}", end="")
                time.sleep(0.1)
                continue
            ego = np.frombuffer(raw_topic.ego).reshape(3)
            nbr = np.frombuffer(raw_topic.nbr).reshape(raw_topic.n_veh, 3)

            vehicles.stack_data(ego[:-1], nbr)
            # left-up side is 0
            X, scene = vehicles.get_inputs(config.EGO_ID, frame)
            if X == -1:
                print("Some vehicles are in same grid. not update")
            else:
                trajectory, veh_idxs, n_veh, scene = to_bytes(X, scene)
                # scenes to bytes
                flag = stub.UpdateGrid(pub_sub_pb2.GridTopic(frame=frame,
                                                             trajectory=trajectory,
                                                             scene=scene,
                                                             veh_ids=veh_idxs,
                                                             n_veh=n_veh,
                                                             ego_id=config.EGO_ID))
                print("elapsed time :", time.time() - st_time, ", UpdateGrid Call", frame)

            frame += config.DOWNSAMPLING_RATE
            #time.sleep(0.1)


if __name__ == "__main__":
    run()
