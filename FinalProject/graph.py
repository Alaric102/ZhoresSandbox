import numpy as np

# Linked node of graph
class Node:
    def __init__(self, state) -> None:
        self.state_ = state
        self.childs = []
        self.parent = None

# Simple graph
class Graph:
    def __init__(self, start, goal) -> None:
        self.start_ = Node(start)
        self.goal_  = Node(goal)