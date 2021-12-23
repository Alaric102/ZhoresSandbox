# https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
# mpirun --allow-run-as-root -n 4 python3 FinalProject/PRRT.py 

# import all the needed modules
from mpi4py import MPI

import math
import random
import time 
import matplotlib.pyplot as plt
import numpy as np

import sys
import resource

sys.setrecursionlimit(10000)

show_animation = False

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
        def __init__(self, rank) -> None:
            self.rank_ = rank

        def Print(self, msg=""):
            print("CPU", self.rank_, ":",  msg, flush=True)

        def Report(self, msg):
            pass

    class Master(Process):
        def __init__(self, rank, size) -> None:
            super().__init__(rank)
            self.size_ = size
            self.Print("Launched " + str(self.size_) +  " processes.")

        def Report(self, msg):
            self.Print(msg)

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
            self.proc = self.Master(rank, size)
        else:
            self.proc = self.Process(rank)
        self.comm.Barrier()

    def draw_graph(self, rnd=None):
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
        plt.pause(0.01)

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

    def SyncGraph(self, new_node):
        data_gather = self.comm.gather(new_node, root=0)

        # msg = ""
        # if type(data_gather) == list:
        #     msg = "data_gather: "
        #     for item in data_gather:
        #         # self.proc.Print(type(item))
        #         if type(item[0]) == self.Node:
        #             msg += "[" + str( round(item[0].x, 5) ) + ", " + str( round(item[0].y, 5) ) + "];"
        # self.proc.Print(msg)

        self.comm.Barrier()

        data_bcast = self.comm.bcast(data_gather, root=0)
        self.comm.Barrier()

        msg = ""
        if type(data_bcast) == list:
            msg = "data_bcast"
            for item in data_bcast:
                if type(item[0]) == self.Node:
                    # msg += "[" + str( round(item[0].x, 5) ) + ", " + str( round(item[0].y, 5) ) + "];"
                    self.node_list.append(item[0])
        # self.proc.Print(msg)
        
        # self.proc.Print( str(len(self.node_list)) )

        self.comm.Barrier()
        return

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]
            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            
            if self.check_if_outside_play_area(new_node, self.play_area) and self.check_collision(new_node, self.obstacle_list):
                new_node = [new_node]
            else:
                new_node = [None]

            self.SyncGraph(new_node)
            # self.proc.Report("Iteration:" + str(i))

            if animation and i % 5 == 0:
                self.draw_graph(rnd_node)

            if self.calc_dist_to_goal(self.node_list[-1].x,
                                      self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.end,
                                        self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

            if animation and i % 5:
                self.draw_graph(rnd_node)

        return None  # cannot find path

def main(gx=6.0, gy=10.0):
    start_time = time.time()
        # ====Search Path with RRT====
    obstacleList = [(5, 5, 1), (3, 6, 2), (3, 8, 2), (3, 10, 2), (7, 5, 2),
                    (9, 5, 2), (8, 10, 1)]  # [x, y, radius]

    # Set Initial parameters
    rrt = RRT( start=[0, 0], goal=[gx, gy], rand_area=[-2, 15], max_iter=10000,expand_dis=0.5, path_resolution=0.1, obstacle_list=obstacleList)

    path = rrt.planning(animation=show_animation)

    duration = time.time() - start_time
    f = open("parallel time collective.dat", "a")
    f.write(str(duration)+'\n')
    f.close()

    if path is None:
        rrt.proc.Report("Cannot find path")
    else:
        rrt.proc.Report("found path!!" + str(duration))

        # Draw final path
        if show_animation:
            rrt.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
            plt.grid(True)
            plt.pause(0.5)  # Need for Mac
            # plt.show()

if __name__ == '__main__':
    main()