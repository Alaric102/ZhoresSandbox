# https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
# mpirun --allow-run-as-root -n 4 python3 FinalProject/PRRT.py 

# import all the needed modules
from typing import Tuple
from mpi4py import MPI
import sys

import math
import random
import time 
import matplotlib.pyplot as plt
import numpy as np

import sys
import resource

# mpi4py.rc.threads = True
# mpi4py.rc.recv_mprobe = True
# mpi4py.rc.fast_reduce = True

# sys.setrecursionlimit(10000)

show_animation = True

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
    
    class Process:
        def __init__(self, rank, size, comm) -> None:
            self.rank_ = rank
            self.size_ = size
            self.comm = comm
            self.updates = []

        def Print(self, msg=""):
            print("CPU", self.rank_, ":",  msg, flush=True)

        def PrintNode(self, node, digits = 5):
            msg = "Not node"
            if type(node) is RRT.Node:
                msg = "[" + str(round(node.x, digits)) + ", " + str(round(node.y, digits)) + "]"
            return msg

        def Report(self, msg):
            pass

        def SendUpdates(self):
            """
            Send new states to Master
            """
            req = self.comm.isend(self.updates, dest=0, tag=self.rank_)
            self.updates = [] # clean updates list

        def GetUpdates(self):
            """
            Receive updates from Master
            """
            req = self.comm.irecv(source = 0, tag = 0)
            try:
                data = req.wait()
            except:
                data = []
            return data
            
    class Master(Process):
        def __init__(self, rank, size, comm) -> None:
            super().__init__(rank, size, comm)
            self.Print("Launched " + str(self.size_) +  " processes.")
            self.updates_dict = {}
            for i in range(0, self.size_):
                self.updates_dict.update( {i : [] } )

        def Report(self, msg):
            self.Print(msg)

        def SendUpdates(self):
            """
            Receive updates from Slaves and hold them
            """
            for i in range(1, self.size_):
                req = self.comm.irecv(source = i, tag = i)
                try:
                    data = req.wait()
                except:
                    data = []
                
                # Add received nodes to updates list of others CPUs
                for item in data:
                    for j in range(0, self.size_):
                        if j != i: self.updates_dict[j].append(item)

            # Add master nodes to updates list
            for item in self.updates:
                for j in range(0, self.size_):
                    if j != 0: self.updates_dict[j].append(item)
            self.updates = [] # clean master updates

        def GetUpdates(self):
            """
            Spread updates among CPUs
            """
            for i in range(1, self.size_):
                req = self.comm.isend(self.updates_dict[i], dest=i, tag=self.rank_)
                self.updates_dict[i] = [] # Clean update list
            
            # procces master updates
            if self.updates_dict[0]:
                local_updates = self.updates_dict[0]
            else:
                local_updates = []
            self.updates_dict[0] = []
            return local_updates

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

        self.comm = MPI.COMM_WORLD
        size = self.comm.Get_size()
        rank = self.comm.Get_rank()
        
        if rank == 0:
            self.proc = self.Master(rank, size, self.comm)
        else: 
            self.proc = self.Process(rank, size, self.comm)
        self.comm.Barrier()

    def draw_graph(self, rnd=None, pause = 0.01):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for (ox, oy, size) in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        if self.play_area is not None:
            plt.plot([self.play_area.xmin, self.play_area.xmax,
                      self.play_area.xmax, self.play_area.xmin,
                      self.play_area.xmin],
                     [self.play_area.ymin, self.play_area.ymin,
                      self.play_area.ymax, self.play_area.ymax,
                      self.play_area.ymin],
                     "-k")

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)
        plt.pause(pause)

    @staticmethod
    def plot_circle(x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)

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
    
    def generate_final_course(self, goal_ind):
        path = [[self.end.x, self.end.y]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

        return path

    def calc_dist_to_goal(self, x, y):
        dx = x - self.end.x
        dy = y - self.end.y
        return math.hypot(dx, dy)

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            if self.check_if_outside_play_area(new_node, self.play_area) and self.check_collision(new_node, self.obstacle_list):
                self.proc.updates.append(new_node)
                self.node_list.append(new_node)

            # msg = "Sended updates: "
            # for item in self.proc.updates:
            #     msg += self.proc.PrintNode(item)
            # self.proc.Print(msg)
            self.proc.SendUpdates()

            new_nodes = self.proc.GetUpdates()
            # msg = "Receive updates: "
            # for item in new_nodes:
            #     msg += self.proc.PrintNode(item)
            # self.proc.Print(msg)
            for item in new_nodes:
                self.node_list.append(item)

            if animation and i % 1 == 0:
                self.draw_graph(rnd_node, pause= 3)
            
            shift = self.proc.rank_ + 1
            shift = 1
            if self.calc_dist_to_goal(self.node_list[-shift].x,
                                        self.node_list[-shift].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-shift], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    self.proc.Print("Found!")
                    
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node, pause=3)

        return None  # cannot find path

def main(gx=6.0, gy=10.0):
        # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]

    # Set Initial parameters
    rrt = RRT( start=[0, 0], goal=[gx, gy], rand_area=[-2, 15], max_iter=1000, expand_dis=0.5, path_resolution=0.1,
        obstacle_list=obstacleList)

    start_time = time.time()
    path = rrt.planning(animation=show_animation)

    duration = time.time() - start_time
    f = open("parallel time.dat", "a")
    f.write(str(duration)+'\n')
    f.close()

    if path is None:
        rrt.proc.Print("Cannot find path")
    else:
        rrt.proc.Print("found path!!:" + str(duration))
        MPI.Finalize()

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(10)  # Need for Mac
            # plt.show()
    sys.exit(1)


if __name__ == '__main__':
    main()
