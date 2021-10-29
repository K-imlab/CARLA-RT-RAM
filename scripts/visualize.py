import grpc
import time
import pygame
import numpy as np
from collections import deque

from messages import pub_sub_pb2_grpc, pub_sub_pb2
import config


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)


class Drawer:
    def __init__(self):
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.font_file = "freesansbold.ttf"
        self.font_size = 20
        self.grid_lat_transformed = 400
        self.grid_lon_transformed = 600
        self.grid_size_lat = 3
        self.grid_size_lon = 13
        self.circle_r = 3
        self.ego_pos_q = deque([], maxlen=int(config.HISTORY_SIZE) // config.DOWNSAMPLING_RATE + 1)

    def draw_history(self, screen, veh_ids, points):
        points = points[:, config.EGO_IDX, :, :]
        for i, j in enumerate(veh_ids):
            color = tuple(np.random.choice(range(256), size=3))
            for lat_lon in points[i]:
                pygame.draw.circle(screen, color, lat_lon, self.circle_r, 1)
            lat = (j // self.grid_size_lon)*(self.grid_lat_transformed // self.grid_size_lat) + (self.grid_lat_transformed // self.grid_size_lat / 2)
            lon = (j % self.grid_size_lon) *(self.grid_lon_transformed // self.grid_size_lon) + (self.grid_lon_transformed // self.grid_size_lon / 2)
            point = (lat,lon)
            font = pygame.font.Font(self.font_file, self.font_size)
            text = font.render(f"{j}", True, color)
            textRect = text.get_rect()
            textRect.center = point
            screen.blit(text, textRect)

    def get_ego_histroy(self, veh_ids, trajectory):
        idx = np.where(veh_ids==config.EGO_ID)[0][0]
        ego_history = trajectory[idx, config.EGO_IDX]
        return ego_history

    def draw_ego_history(self, screen, ego_history):
        history = ego_history[config.EGO_IDX]
        center = [self.grid_lat_transformed//2, self.grid_lon_transformed//2]
        t_p = center - history

        font = pygame.font.Font(self.font_file, self.font_size)
        text = font.render(f"{config.EGO_ID}", True, self.BLUE)
        textRect = text.get_rect()
        textRect.center = history[0]
        screen.blit(text, textRect)

        for point in t_p:
            pygame.draw.circle(screen, self.BLUE, point, self.circle_r, 5)

    def trans_to_imloc(self, ego_info, veh_ids, trajectory):
        lat, lon, heading = ego_info  # radian
        idx = np.where(veh_ids == config.EGO_ID)[0][0]
        ego_view = trajectory[idx]  # (39,16,2)

        rotate_mat = np.array([[np.cos(heading), -np.sin(heading)],
                               [np.sin(heading),  np.cos(heading)]])

        trans_ego_view = np.matmul(rotate_mat, ego_view)
        trans_ego_view = trans_ego_view * 5

        return trans_ego_view


def from_bytes(raw_topic=None, grid_topic=None, pred_topic=None, risk_topic=None):
    try:
        t = raw_topic.frame
        image = np.frombuffer(raw_topic.image, dtype=np.uint8).reshape((400, 600, 3))
        ego_info = np.frombuffer(raw_topic.ego)
        cam_loc = np.frombuffer(raw_topic.cam_loc)

        print(f"raw topic {cam_loc} {ego_info}")
    except:
        t = None
        image = None
        ego_info = None
        nbr_info = None
        pass

    try:
        grid_topic_t = grid_topic.frame
        print(f"grid topic {grid_topic_t}")
        ego_id = grid_topic.ego_id
        scene = np.frombuffer(grid_topic.scene).reshape([config.GRID_SIZE[0], config.GRID_SIZE[1]])
        trajectory = np.frombuffer(grid_topic.trajectory).reshape(
            [-1, config.GRID_SIZE[0]*config.GRID_SIZE[1],
             int(config.HISTORY_SIZE) // config.DOWNSAMPLING_RATE + 1, 2])
        veh_ids = scene[np.where(scene != 0)]

    except Exception as e:
        print(e)
        ego_id = None; veh_ids = None; scene = None; trajectory = None
        pass

    try:
        pred_trajectory = np.frombuffer(pred_topic.pred_trajectory).reshape(
            [-1, int(config.FUTURE_SIZE) // config.DOWNSAMPLING_RATE, 2]
        )
    except:
        pred_trajectory=None
        pass

    try:
        n_collision = risk_topic.n_collision
        risk_2darray = np.frombuffer(risk_topic.risk).reshape([n_collision, 3])
        # risks : [(collision_time, id, exp_ttc), ......]
    except:
        n_collision=None; risk_2darray=None
        pass

    return (t, image, ego_info, cam_loc), (ego_id, veh_ids, scene, trajectory), pred_trajectory, risk_2darray


def run():
    drawer = Drawer()
    frame = 0
    pygame_fps = 30
    pygame.init()
    display = pygame.display.set_mode((400, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Bird Eye View")
    pygame_clock = pygame.time.Clock()

    with grpc.insecure_channel("localhost:50055") as channel:
        stub = pub_sub_pb2_grpc.VisualizeStub(channel)
        while True:

            try:
                st_time = time.time()
                flag_topic = stub.CheckFrame(pub_sub_pb2.TimeRequest(frame=frame))
                flag = flag_topic.flag
            except:
                continue
            if flag == 1:
                try:
                    raw_topic = stub.GetHistory(pub_sub_pb2.TimeRequest(frame=frame))
                except:
                    raw_topic = None
                    pass
                try:
                    grid_topic = stub.GetGrid(pub_sub_pb2.TimeRequest(frame=frame))
                except:
                    grid_topic = None
                    pass
                try:
                    pred_topic = stub.GetPred(pub_sub_pb2.TimeRequest(frame=frame))
                except:
                    pred_topic = None
                    pass
                try:
                    risk_topic = stub.GetRisk(pub_sub_pb2.TimeRequest(frame=frame))
                except:
                    risk_topic = None
                    pass
            elif flag == 0:
                print("\r: There are no ego vehicles yet in", frame, end="")
                frame += config.DOWNSAMPLING_RATE
                time.sleep(0.1)
                continue
            elif flag == 2:
                print(f"\r: Waiting for every process {frame}", end="")
                time.sleep(0.1)
                continue
            else:
                print(f"\r: It was omitted. {frame}", end="")
                frame += config.DOWNSAMPLING_RATE
                time.sleep(0.1)
                continue

            ## BODY ##

            raw, grid, pred, risk = from_bytes(raw_topic, grid_topic, pred_topic, risk_topic)
            t, image, ego_info, cam_loc = raw
            ego_id, veh_ids, scene, trajectory = grid

            pygame_clock.tick_busy_loop(pygame_fps)
            if image is not None:
                surface = pygame.surfarray.make_surface(image)
                display.blit(surface, (0,0))

                # draw grid and notate veh_id draw history
                if grid_topic is not None:
                    trans_ego = drawer.trans_to_imloc(ego_info, veh_ids, trajectory)
                    # ego_history = drawer.get_ego_histroy(veh_ids, trajectory)
                    drawer.draw_ego_history(display, trans_ego)

                    # idx_of_ego = np.where(veh_ids == ego_id)[0][0]
                    # if len(trajectory) >= 1:
                    #     np.save("scene.npy", scene)
                    #     np.save("veh_ids.npy", veh_ids)
                    #     np.save("traj.npy", trajectory)
                    # drawer.draw_history(display, veh_ids, trajectory)  # ego [n_veh, 39,16,2]

                    pass

                # draw predict
                if pred_topic is not None:
                    pass

                # draw risk
                if risk_topic is not None:
                    pass

                # print("elapsed time : ", time.time() - st_time, "frame : ", frame)
                frame += config.DOWNSAMPLING_RATE
                pygame.display.flip()
                pygame.event.pump()

            # time.sleep(0.1)





def pygame_tut():
    pygame.init()
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)

    size = [400, 600]
    screen = pygame.display.set_mode(size)

    pygame.display.set_caption("imlab")

    done = False
    clock = pygame.time.Clock()

    point = np.array([200., 200.])
    points = [point]
    for i in range(15):
        dlat = np.random.rand()
        dlon = np.random.rand()
        point = point - np.array([dlat, dlon])
        points.append(point)
    points = np.array(points)

    while not done:
        clock.tick(10)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        screen.fill(WHITE)
        pygame.draw.line(screen, GREEN, [0,0], [100,150], 5)
        pygame.draw.line(screen, BLUE, [0, 300], [100, 150], 5)

        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    run()
    # pygame_tut()

