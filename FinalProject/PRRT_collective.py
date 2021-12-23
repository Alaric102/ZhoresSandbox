# https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
# mpirun --allow-run-as-root -n 4 python3 FinalProject/PRRT.py 

# import all the needed modules
from mpi4py import MPI

from mpi4py import MPI

# regular process
class Process:
    def __init__(self, rank) -> None:
        self.rank_ = rank

    def Print(self, msg=""):
        print("CPU", self.rank_, ":",  msg, flush=True)

    def Sync(self, data = None):
        pass

# main process
class Master(Process):
    def __init__(self, rank, size) -> None:
        super().__init__(rank)
        self.size_ = size
        self.Print("Launched " + str(self.size_) +  " processes.")
    
    def Sync(self, data = None):
        # data = comm.bcast(data, root=0)
        pass
        # for i in range(1, self.size_):
        #     self.Print("Broadcas")

import math
import random
import time 
import matplotlib.pyplot as plt
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

comm.Barrier()

if rank == 0:
    proc = Master(rank, size)
else:
    proc = Process(rank)

class RRT:
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y
            self.path_x = []
            self.path_y = []
            self.parent = None

    class AreaBounds:
        def __init__(self, area):
            self.xmin = float(area[0])
            self.xmax = float(area[1])
            self.ymin = float(area[2])
            self.ymax = float(area[3])
    
    def __init__(self, start, goal, obstacle_list, rand_area,
            expand_dis=0.05, path_resolution=0.01, goal_sample_rate=5, 
            max_iter=10000, play_area=None ):
        """
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        play_area:stay inside this area [xmin,xmax,ymin,ymax]
        """
        self.start = self.Node(start[0], start[1])
        self.end = self.Node(goal[0], goal[1])
        self.min_rand = rand_area[0]
        self.max_rand = rand_area[1]
        if play_area is not None:
            self.play_area = self.AreaBounds(play_area)
        else:
            self.play_area = None

        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []

    def get_random_node(self):
        if random.randint(0, 100) > self.goal_sample_rate:
            rnd = self.Node(
            random.uniform(self.min_rand, self.max_rand),
            random.uniform(self.min_rand, self.max_rand))
        else:  # goal point sampling
            rnd = self.Node(self.end.x, self.end.y)
        return rnd
    
    @staticmethod
    def calc_distance_and_angle(from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)

        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)
        
        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y

        new_node.parent = from_node
        return new_node

    @staticmethod
    def get_nearest_node_index(node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    @staticmethod
    def check_if_outside_play_area(node, play_area):
        if play_area is None: return True  # no play_area was defined, every pos should be ok

        if node.x < play_area.xmin or node.x > play_area.xmax or \
           node.y < play_area.ymin or node.y > play_area.ymax:
            return False  # outside - bad
        else:
            return True  # inside - ok

    @staticmethod
    def check_collision(node, obstacleList):
        if node is None: return False

        for (ox, oy, size) in obstacleList:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]

            if min(d_list) <= size**2:
                return False  # collision
        return True  # safe

    def SyncGraph(self, new_node):
        data = comm.gather(new_node, root=0)
        data = comm.bcast(data, root=0)
        [self.node_list.append(item) for item in data]
        
        # if proc.rank_ == 0:
        #     for item in self.node_list:
        #             proc.Print("[" + str(round(item.x, 3)) + ", " + str(round(item.y, 3)) + "]")
        #     proc.Print()
        # comm.Barrier()

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            
            # proc.Print("msg2: [" + str(round(nearest_node.x, 1)) + ", " + str(round(nearest_node.y, 1)) + "] --> [" +
            #     str(round(new_node.x, 3)) + ", " + str(round(new_node.y, 3)) + "]")

            if self.check_if_outside_play_area(new_node, self.play_area) and self.check_collision(new_node, self.obstacle_list):
                self.SyncGraph(new_node)
                # self.node_list.append(new_node)

        #     if animation and i % 5 == 0:
        #         self.draw_graph(rnd_node)

        #     if self.calc_dist_to_goal(self.node_list[-1].x,
        #                               self.node_list[-1].y) <= self.expand_dis:
        #         final_node = self.steer(self.node_list[-1], self.end,
        #                                 self.expand_dis)
        #         if self.check_collision(final_node, self.obstacle_list):
        #             return self.generate_final_course(len(self.node_list) - 1)

        #     if animation and i % 5:
        #         self.draw_graph(rnd_node)

        return None  # cannot find path

def main(gx=6.0, gy=10.0):

    # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]
    
    # Set Initial parameters
    rrt = RRT( start=[0, 0], goal=[gx, gy], rand_area=[-2, 15], max_iter=10000,expand_dis=0.5, path_resolution=0.1,
        obstacle_list=obstacleList)

    start_time = time.time()
    path = rrt.planning(animation=True)

    duration = time.time() - start_time
    f = open("parallel time collective.dat", "a")
    f.write(str(duration)+'\n')
    f.close()

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!", duration)

    #     # Draw final path
    #     if show_animation:
    #         rrt.draw_graph()
    #         plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    #         plt.grid(True)
    #         plt.pause(0.5)  # Need for Mac
    #         # plt.show()

if __name__ == '__main__':
    main()