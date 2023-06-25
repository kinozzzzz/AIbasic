import numpy as np
from typing import List
from utils import TreeNode
from simuScene import PlanningMap


### 定义一些你需要的变量和函数 ###
STEP_DISTANCE = 0.35
STEP_LONG_DISTANCE = 3.5
TARGET_THREHOLD = 0.25
debug = 0

### 定义一些你需要的变量和函数 ###


class RRT:
    def __init__(self, walls) -> None:
        """
        输入包括地图信息，你需要按顺序吃掉的一列事物位置
        注意：只有按顺序吃掉上一个食物之后才能吃下一个食物，在吃掉上一个食物之前Pacman经过之后的食物也不会被吃掉
        """
        self.map = PlanningMap(walls)
        self.walls = walls
        
        self.height = self.map.height
        self.width = self.map.width
        
        
    def find_path(self, current_position, next_food):
        """
        在程序初始化时，以及每当 pacman 吃到一个食物时，主程序会调用此函数
        current_position: pacman 当前的仿真位置
        next_food: 下一个食物的位置
        
        本函数的默认实现是调用 build_tree，并记录生成的 path 信息。你可以在此函数增加其他需要的功能
        """
              

        self.path = self.build_tree(current_position, next_food)
        if debug != 0:
            for i in self.path:
                print("target pos:",i)
            print()
        
    def get_target(self, current_position, current_velocity):
        """
        主程序将在每个仿真步内调用此函数，并用返回的位置计算 PD 控制力来驱动 pacman 移动
        current_position: pacman 当前的仿真位置
        current_velocity: pacman 当前的仿真速度
        一种可能的实现策略是，仅作为参考：
        (1)记录该函数的调用次数
        (2)假设当前 path 中每个节点需要作为目标 n 次
        (3)记录目前已经执行到当前 path 的第几个节点，以及它的执行次数，如果超过 n，则将目标改为下一节点
        
        你也可以考虑根据当前位置与 path 中节点位置的距离来决定如何选择 target
        
        同时需要注意，仿真中的 pacman 并不能准确到达 path 中的节点。你可能需要考虑在什么情况下重新规划 path
        """
        target_pose = self.path[-1]
        if (np.sqrt(np.sum(np.square(current_position-target_pose))) < 0.5
            and len(self.path) > 1):
            self.path.pop()
            target_pose = self.path[-1]
        s = current_position + 2*(target_pose - current_position) - 0.1*current_velocity
        # print(target_pose)
        return s
        
    def build_tree(self, start, goal):
        """
        实现你的快速探索搜索树，输入为当前目标食物的编号，规划从 start 位置食物到 goal 位置的路径
        返回一个包含坐标的列表，为这条路径上的pd targets
        你可以调用find_nearest_point和connect_a_to_b两个函数
        另外self.map的checkoccupy和checkline也可能会需要，可以参考simuScene.py中的PlanningMap类查看它们的用法
        """
        if_hit,_ = self.map.checkline(start.tolist(),goal.tolist())
        if not if_hit:
            path = self.straight_move(start,goal)
            return path
        
        path = []
        graph: List[TreeNode] = []
        graph.append(TreeNode(-1, start[0], start[1]))
        while(True):
            if np.random.rand() > 0.65:
                point = goal
            else:
                point = self.sample()
            nearest_idx, nearest_point = self.find_nearest_point(point,graph)
            if_empty, new_point = self.connect_a_to_b(nearest_point,point)
            if if_empty:
                new_node = TreeNode(nearest_idx,new_point[0],new_point[1])
                graph.append(new_node)
                if_hit,_ = self.map.checkline(new_point.tolist(),goal.tolist())
                if not if_hit:
                    # print("final point",new_point)
                    break
        
        if debug == 2:
            for node in graph:
                print("node",node.pos)
        path = self.straight_move(graph[-1].pos,goal)
        path.extend(self.build_tree(start,graph[-1].pos))
        '''
        idx = -1
        while idx != 0:
            path.append(graph[idx].pos)
            idx = graph[idx].parent_idx
        '''
        return path

    @staticmethod
    def find_nearest_point(point, graph: List[TreeNode]):
        """
        找到图中离目标位置最近的节点，返回该节点的编号和到目标位置距离
        输入：
        point：维度为(2,)的np.array, 目标位置
        graph: List[TreeNode]节点列表
        输出：
        nearest_idx, nearest_distance 离目标位置距离最近节点的编号和距离
        """
        nearest_idx = -1
        nearest_distance = 10000000.
        nearest_point = np.zeros(2)
        for idx,node in enumerate(graph):
            distance = np.sqrt(np.sum(np.square(point-node.pos)))
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_idx = idx
                nearest_point = node.pos
        return nearest_idx, nearest_point
    
    def connect_a_to_b(self, point_a, point_b):
        """
        以A点为起点，沿着A到B的方向前进STEP_DISTANCE的距离，并用self.map.checkline函数检查这段路程是否可以通过
        输入：
        point_a, point_b: 维度为(2,)的np.array，A点和B点位置，注意是从A向B方向前进
        输出：
        is_empty: bool，True表示从A出发前进STEP_DISTANCE这段距离上没有障碍物
        newpoint: 从A点出发向B点方向前进STEP_DISTANCE距离后的新位置，如果is_empty为真，之后的代码需要把这个新位置添加到图中
        """
        direction = point_b - point_a
        newpoint = point_a + direction / np.sqrt(np.sum(np.square(direction))) * STEP_DISTANCE
        if_hit,hit_position =self.map.checkline(point_a.tolist(),newpoint.tolist())
        return not if_hit, newpoint

    def sample(self):
        while(True):
            height = np.random.uniform(0,self.height)
            width = np.random.uniform(0,self.width)
            point = np.array([height,width])
            if not self.map.checkoccupy(point):
                return point
    
    def straight_move(self,point_a,point_b):
        path = []
        len = np.sqrt(np.sum(np.square(point_a - point_b)))
        direction = (point_b - point_a) / len
        while(True):
            if(len < STEP_LONG_DISTANCE):
                path.append(point_b)
                path.reverse()
                return path
            new_point = point_a + direction * STEP_LONG_DISTANCE
            path.append(new_point)
            point_a = new_point
            len -= STEP_LONG_DISTANCE
            
        