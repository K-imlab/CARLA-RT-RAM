from messages import pub_sub_pb2_grpc, pub_sub_pb2

import grpc
import numpy as np
import os
import time


def read_txt(path):
    with open(path) as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [list(map(float, x.strip().split(','))) for x in content]

    return content


class Sender:
    def __init__(self, path):
        self.ego_path = os.path.join("D:\\HJKIM\\RT-RAM-grpc","fake_data", 'ego_sample.csv')
        self.nbr_path = os.path.join("D:\\HJKIM\\RT-RAM-grpc","fake_data", 'nbr_sample.csv')
        self.ego_content = np.array(read_txt(self.ego_path))
        self.nbr_content = np.array(read_txt(self.nbr_path))
        self.frame_list = self.get_frame_list()
        self.idx = 0
        self.len = len(list(self.frame_list))
        self.idx_ego = 0
        self.idx_nbr = 0

    def get_frame_list(self):
        if len(np.unique(self.ego_content[:,0])) == len(np.unique(self.nbr_content[:,0])) :
            return np.unique(self.ego_content[:,0])
        else :
            return -1

    def __len__(self):
        return self.len

    def send(self):
        ego_data = []
        nbr_data = []
        if self.idx < self.len:
            frame = self.frame_list[self.idx]
            while self.ego_content[self.idx_ego][0] == frame:
                ego_data = self.ego_content[self.idx_ego][1:]
                self.idx_ego += 1

            while self.nbr_content[self.idx_nbr][0] == frame:
                nbr_data.append(self.nbr_content[self.idx_nbr][1:])
                self.idx_nbr += 1

            self.idx += 1

            return frame, np.array(ego_data), np.array(nbr_data)

        else:
            print("Out of stock")
            return -1


def run():
    sender = Sender(".")

    with grpc.insecure_channel("localhost:50054") as channel:
        while True:
            t, ego, nbr = sender.send()
            frame = int(t * 10)
            n_veh = len(nbr)
            print(t, ego, nbr)
            ego = ego.tobytes()
            nbr = nbr.tobytes()
            stub = pub_sub_pb2_grpc.RawDataStub(channel)
            flag = stub.UpdateHistory(pub_sub_pb2.RawTopic(frame=frame,
                                                           ego=ego,
                                                           nbr=nbr,
                                                           n_veh=n_veh))
            print("UpdateHistory Call", frame)
            time.sleep(0.1)


if __name__ == "__main__":
    run()
