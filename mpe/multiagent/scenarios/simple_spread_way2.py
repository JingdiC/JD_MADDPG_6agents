import numpy as np
from mpe.multiagent.core import World, Agent, Landmark
from mpe.multiagent.scenario import BaseScenario

"""
N agents, N landmarks. 
Agents are rewarded based on how far any agent is from its target landmark. 
Agents are penalized if they collide with other agents. 
So, agents have to learn to cover its own target landmark while avoiding collisions.

fully obs and fully comm
use with train_way3 to find the correlation inside the 22 num of obs vector of agent 1

comm cost added, bottleneck used in train way3

"""
class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 4
        num_agents = 3
        num_landmarks = 3
        world.collaborative = False
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = False
            agent.size = 0.15
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for landmarks
        world.landmarks[0].color = np.array([0.65,0.15,0.15])
        world.landmarks[1].color = np.array([0.15,0.65,0.15])
        world.landmarks[2].color = np.array([0.15,0.15,0.65])

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])

        # assign goals to agents
        for agent in world.agents:
            agent.goal = None

        # want agents to go to the goal landmark
        for i, agent in enumerate(world.agents):
            agent.goal = np.random.choice(world.landmarks)
            agent.goal.color = agent.goal.color + np.array([0.45, 0.45, 0.45])

        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        arrived_agents = 0
        min_dists = 0
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal.state.p_pos))) for a in world.agents]
        min_dists += min(dists)
        rew -= min(dists)
        if min(dists) < 0.1:
                arrived_agents += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, arrived_agents)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world, i):
        # Agents are rewarded based on minimum agent distance to a target landmark, penalized for collisions
        # how to choose a target landmark

        comm_cost = [0.2,0.2,0.3,0.3,0.5,0.5]
        alpha = 0.05
        rew = -alpha * (comm_cost[i[0]] + comm_cost[i[1]])
        dists = [np.sqrt(np.sum(np.square(a.state.p_pos - a.goal.state.p_pos))) for a in world.agents]
        rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)