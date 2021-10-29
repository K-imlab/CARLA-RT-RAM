import grpc
import numpy as np
from shapely.geometry import Polygon
import time

import config
from messages import pub_sub_pb2_grpc, pub_sub_pb2


def get_ttc(ego_collision_t, ego_collision_t1, nbr_collision_t, alpha=0.5):
    '''
    ttc = range / velocity
    ego_collision_t (dtype : numpy, shape : (,2))
    nbr_collision_t (dtype : numpy, shape : (,2))
    '''

    distance = np.sqrt(np.sum((ego_collision_t - nbr_collision_t) ** 2))
    velocity = np.sqrt(np.sum((ego_collision_t - ego_collision_t1) ** 2))

    ttc = distance / velocity
    exp_ttc = np.exp(-alpha*ttc**2)
    return exp_ttc


def get_rectangle(latlon_t, latlon_t1, veh_width, veh_length, safe_dis):
    theta = np.arctan((latlon_t1[0] - latlon_t[0])/ (latlon_t1[1] - latlon_t[1]))

    rec = np.array([[-veh_width/2,-veh_length/2],
                    [-veh_width/2, veh_length/2+safe_dis],
                    [veh_width/2, veh_length/2+safe_dis],
                    [veh_width/2, -veh_length/2]])

    rotate = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    rotate_rec = np.matmul(rec, rotate)
    rotate_rec[:, 0] = rotate_rec[:, 0] + latlon_t[0]
    rotate_rec[:, 1] = rotate_rec[:, 1] + latlon_t[1]

    return rotate_rec


def get_collision_risk(ego_pred, nbr_pred, nbr_ids):
    '''
    if rectangle of vehicles have intersection, that is collision
    :param ego_pred: (25,2)
    :param nbr_pred: (n_nbr, 25,2)
    :param nbr_ids: (n_nbr,)
    :return: sequence of (collision time (0~25), with id, ttc)
    '''
    lt = ego_pred[:-1]
    lt1 = ego_pred[1:]
    risks = []
    for latlon_t, latlon_t1 in zip(lt, lt1):
        t = 0
        ego_rec_t = get_rectangle(latlon_t, latlon_t1,
                                  config.WIDTH, config.LENGTH,
                                  config.SAFE_DIS)
        p = Polygon(ego_rec_t)
        for i, nbr_id in enumerate(nbr_ids):
            nbr_rec_t = get_rectangle(nbr_pred[i, t], nbr_pred[i, t+1],
                                      config.WIDTH, config.LENGTH,
                                      config.SAFE_DIS)
            p1 = Polygon(nbr_rec_t)
            is_collision = p.intersection(p1)
            if is_collision:
                collision_time = t
                collision_nbr_id = nbr_id
                ego_collision_t = ego_pred[collision_time]
                ego_collision_t1 = ego_pred[collision_time + 1]
                nbr_collision_t = nbr_pred[i, collision_time]
                ttc = get_ttc(ego_collision_t, ego_collision_t1, nbr_collision_t)
                risk = (collision_time, collision_nbr_id, ttc)
                print(f"collision detect!, {risk}")
                risks.append(risk)

        t += 1

    return np.array(risks)


def from_bytes(pred_topic):
    frame = pred_topic.frame
    pred_trajectory = np.frombuffer(pred_topic.pred_trajectory)
    veh_ids = np.frombuffer(pred_topic.veh_ids, dtype=np.int32)
    scene = np.frombuffer(pred_topic.scene)
    n_veh = pred_topic.n_veh
    ego_i = pred_topic.ego_i

    pred_trajectory = pred_trajectory.reshape([n_veh,
                                               int(config.FUTURE_SIZE) // config.DOWNSAMPLING_RATE, 2])
    scene = scene.reshape([config.GRID_SIZE[0], config.GRID_SIZE[1]])
    nbr_pred = np.delete(pred_trajectory, ego_i, axis=0)
    ego_pred = pred_trajectory[ego_i]
    nbr_ids = np.delete(veh_ids, ego_i)
    ego_veh_id = veh_ids[ego_i]

    return ego_pred, nbr_pred, nbr_ids, int(ego_veh_id)


def run():
    frame = 0
    with grpc.insecure_channel("localhost:50055") as channel:
        stub = pub_sub_pb2_grpc.RiskAssessmentStub(channel)
        while True:
            st_time = time.time()
            flag_topic = stub.CheckFrame(pub_sub_pb2.TimeRequest(frame=frame))
            flag = flag_topic.flag
            if flag == 1:
                pred_topic = stub.GetPred(pub_sub_pb2.TimeRequest(frame=frame))
            elif flag == 0:
                print("\r: There are no ego vehicles yet in", frame, end="")
                frame += config.DOWNSAMPLING_RATE
                time.sleep(0.1)
                continue
            elif flag == 2:
                print(f"\r: Waiting for trajectory prediction {frame}", end="")
                time.sleep(0.1)
                continue
            else:
                print(f"\r: It was omitted from the network_forward process. {frame}", end="")
                frame += config.DOWNSAMPLING_RATE
                time.sleep(0.1)
                continue

            ego_pred, nbr_pred, nbr_ids, ego_id = from_bytes(pred_topic)
            risks_ndarray = get_collision_risk(ego_pred, nbr_pred, nbr_ids)
            number_of_collision = len(risks_ndarray)
            # risks : [(collision_time, id, exp_ttc), ...]

            stub.UpdateRisk(pub_sub_pb2.RiskTopic(frame=frame,
                                                  risk=risks_ndarray.tobytes(),
                                                  n_collision=number_of_collision,
                                                  ego_id=ego_id))
            print("elapsed time : ", time.time() - st_time, "UpdateRisk", frame)
            frame += config.DOWNSAMPLING_RATE
            #time.sleep(0.1)


if __name__ == "__main__":
    run()
