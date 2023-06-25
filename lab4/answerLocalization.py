from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 0.75
MAX_ERROR = 50000
Position_Gauss = 0.09
View_Gauss = 0.09

def if_allow(line,column,walls):
    for node in walls:
        if np.abs(line - node[0]) >= COLLISION_DISTANCE or np.abs(column - node[1]) >= COLLISION_DISTANCE:
            continue
        return 0
    return 1

def get_idx(probability,p):
    for idx,prob in enumerate(probability):
        if p < prob:
            return idx
            
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    line_idx = np.max(walls[:,0])
    column_idx = np.max(walls[:,1])
    i = 0
    while i < N:
        line = np.random.uniform(0,line_idx)
        column = np.random.uniform(0,column_idx)
        if(if_allow(line,column,walls)):
            theta = np.random.uniform(-np.pi,np.pi)
            all_particles.append(Particle(line,column,theta,1/N))
            i += 1
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    ### 你的代码 ###
    # weight = np.exp(-weight_K * np.sum(np.abs(estimated-gt)))
    '''
    if resample_times <= 16:
        weight_K = 0.39
    else:
        weight_K = 9.5
    #weight_K = 9.5
    '''
    weight_K = 1.1
    weight = np.exp(-weight_K * np.linalg.norm(estimated-gt))
    
    ### 你的代码 ##
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入： 
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles = []
    probability = [particles[0].weight]
    for particle in particles[1:]:
        probability.append(probability[-1] + particle.weight)
    i = 0
    while i < len(particles)-100:
        p = np.random.uniform(0,probability[-1])
        idx = get_idx(probability,p)
        particle:Particle = particles[idx]
        gener_times = 0
        while(True):
            if(gener_times >= 10):
                i -= 1
                break
            dx = np.random.normal(0,Position_Gauss)
            dy = np.random.normal(0,Position_Gauss)
            new_x = dx + particle.position[0]
            new_y = dy + particle.position[1]
            if(if_allow(new_x,new_y,walls)
               and if_allow(particle.position[0]+dx/2,particle.position[1]+dy/2,walls)
               ):
                #print(idx,end=" ")
                new_theta = np.random.normal(0,View_Gauss) + particle.theta
                new_particle = Particle(new_x,new_y,new_theta,particle.weight)
                resampled_particles.append(new_particle)
                break
            gener_times += 1
        i += 1
    resampled_particles.extend(generate_uniform_particles(walls,100))
    '''
    N = len(particles)
    for particle in particles:
        for _ in range(int(N * particle.weight)):
            new_x = np.random.normal(0,0.09) + particle.position[0]
            new_y = np.random.normal(0,0.09) + particle.position[1]
            if if_allow(new_x,new_y,walls):
                new_theta = np.random.normal(0,0.01) + particle.theta
                new_particle = Particle(new_x,new_y,new_theta,1/N)
                resampled_particles.append(new_particle)
    if len(resampled_particles) != N:
        resampled_particles.extend(generate_uniform_particles(walls,N-len(resampled_particles)))
    '''
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    p.theta += dtheta
    p.position[0] += traveled_distance * np.cos(p.theta)
    p.position[1] += traveled_distance * np.sin(p.theta)
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    ### 你的代码 ###
    #for particle in particles:
        #print(particle.position)
    # print()
    new_particles = particles[0:3]
    average_x = np.mean([particle.position[0] for particle in new_particles])
    average_y = np.mean([particle.position[1] for particle in new_particles])
    average_theta = np.mean([particle.theta for particle in new_particles])
    return Particle(average_x,average_y,average_theta,0)
    ### 你的代码 ###