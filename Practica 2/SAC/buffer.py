import torch
from numpy.random import randint


class Memory:
    def __init__(self, memory_size: int, state_size: tuple):
        """
        Memoria utilizada para guardar la experiencia del entorno
        """
        self.memory_size = memory_size
        self.current_index = 0
        self.state = torch.zeros(size=(memory_size, *state_size), dtype=torch.uint8)
        self.state_ = torch.zeros(size=(memory_size, *state_size), dtype=torch.uint8)
        self.action = torch.zeros(memory_size, dtype=torch.int8)
        self.reward = torch.zeros(memory_size, dtype=torch.int)
        self.terminal = torch.zeros(memory_size, dtype=torch.bool)

        self.objetive = torch.zeros(memory_size, dtype=torch.bool)
        self.area = torch.zeros(size=(memory_size, *state_size), dtype=torch.uint8)

    def store_sars_(self, s, a, r, s_, done,obj,area):
        index = self.current_index % self.memory_size
        self.state[index] = s
        self.state_[index] = s_
        self.action[index] = a
        self.reward[index] = r
        self.terminal[index] = done
        self.objetive = obj
        self.current_index += 1
        self.area = area
    def sample(self, batch_size: int):
        batch = randint(min(self.current_index, self.memory_size), size=batch_size)
        a_batch = self.action[batch]
        s_batch = self.state[batch]
        s_batch_ = self.state_[batch]
        r_batch = self.reward[batch]
        t_batch = self.terminal[batch]
        return s_batch, a_batch, r_batch, s_batch_, t_batch
