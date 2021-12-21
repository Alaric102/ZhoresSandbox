# https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
# mpirun --allow-run-as-root -n 4 python3 FinalProject/PPRRT.py 

# import all the needed modules
from mpi4py import MPI
from proc_shell import Master, Process

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

    def planning(self, animation=True):
        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.get_random_node()

            proc.print( "rnd_node: " + str(round(rnd_node.x, 1)) + ", " + str(round(rnd_node.y, 1)) )

            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
        #     nearest_node = self.node_list[nearest_ind]

        #     new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

        #     if self.check_if_outside_play_area(new_node, self.play_area) and \
        #        self.check_collision(new_node, self.obstacle_list):
        #         self.node_list.append(new_node)

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
    rrt = RRT( start=[0, 0], goal=[gx, gy], rand_area=[-2, 15], max_iter=10,
        obstacle_list=obstacleList)

    path = rrt.planning(animation=True)

    # duration = time.time() - start_time
    # f = open("parallel time.dat", "a")
    # f.write(str(duration)+'\n')
    # f.close()

    # if path is None:
    #     print("Cannot find path")
    # else:
    #     print("found path!!")

    #     # Draw final path
    #     if show_animation:
    #         rrt.draw_graph()
    #         plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    #         plt.grid(True)
    #         plt.pause(0.5)  # Need for Mac
    #         # plt.show()

if __name__ == '__main__':
    main()
