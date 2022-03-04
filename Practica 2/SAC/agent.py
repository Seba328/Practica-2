
import torch
import cv2
import numpy as np
import buffer as bf
import random
from torch.utils.tensorboard import SummaryWriter #Libreria utilizada para generar grafico tensor


class AtariAgent:
    def __init__(self, action_space: np.array, memory_par: tuple, game: tuple, reward_clip: bool):
        """
        Inicializaciones.
        """
        self.game_name, self.environment= game
        self.reward_clip = reward_clip

        dir = 'SAC/Resultados/'
        self.writer = SummaryWriter(dir)
        self.gamma = 0.99
        self.epsilon, self.epsilon_decay, self.mini_epsilon, self.final_epsilon = 0.1, 5e-6, 0.05, 0.01
        self.learn_start_step = 20000
        self.learn_cur, self.learn_replace, self.learn_interval = 0, 1000, 4
        self.no_op_num = 7
        self.episodes = 100000
        self.explore_frame = 5e6
        self.action_space, self.action_space_len = action_space, len(action_space)
        self.frames_num = memory_par[1][0]
        self.multi_frames_size, self.single_img_size = memory_par[1], (1, *memory_par[1][1:])
        self.memory = bf.Memory(*memory_par)
        self.multi_frames = torch.zeros(size=memory_par[1], dtype=torch.float32)
        self.scores = np.zeros(self.episodes, dtype=np.float16)
        self.batch_size = 32
        self.frames_count = 0
        self.step_num, self.step_count = 4, 0
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.global_img = []
        self.gen = None
        self.opciones = 1,2,3,4
        self.area = None
        self.load_img()

    def find_objects(self,board, threshold=0.90):
        """
        Funcion utilizada para encontrar los elementos dentro del entorno Ms-Pacman
        """
        obj_centroids={}
        i = 0
        for obj in self.global_img:
            if obj[0] not in obj_centroids: 
                obj_centroids[obj[0]] = []
            res = cv2.matchTemplate(board, obj[1], cv2.TM_CCOEFF_NORMED)
            loc = np.argwhere(res >= threshold)

            w, h = obj[1].shape[::-1]
            for o in loc:
                y, x = o
                centroid = (x + (w / 2), y + (h / 2))
                obj_centroids[obj[0]].append(list(centroid))
            i+=1
        return obj_centroids

    def isArea(self,observation,height,width):
        """
        Funcion utilizada para retornar el area asignada para el desplazamiento del agente.
        """
        objetive = False
        if self.area == 1:
            area_1 = observation[0:height-170,0:width-280]
            objects = self.find_objects(area_1)
            if objects['agent']:
                objetive = True
                self.area = next(self.gen)
            return objetive,area_1
        elif self.area == 2:
            area_2 = observation[120:height-40,0:width-280]
            objects = self.find_objects(area_2)
            if objects['agent']:
                objetive = True
                self.area = next(self.gen)
            return objetive,area_2
        elif self.area == 3:
            area_3 = observation[0:height-170,120:width]
            objects = self.find_objects(area_3)
            if objects['agent']:
                objetive = True
                self.area = next(self.gen)
            return objetive,area_3
        elif self.area == 4:
            area_4 = observation[120:height-40,120:width]
            objects = self.find_objects(area_4)
            if objects['agent']:
                objetive = True
                self.area = next(self.gen)
            return objetive,area_4

        
    def aleatorio(self,opciones):
        """
        Funcion que selecciona un elemento random sin resposicion.
        """
        opciones = list(opciones)
        while True:
            r = random.choice(opciones)
            opciones.remove(r)
            yield r
    def preprocess_observation(self, observation: np.array) -> torch.Tensor:
        """
        Se preprocesa la imagen del entorno.
        """
        image = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        height, width = image.shape[0:2]
        objetive,area = self.isArea(image,height,width)
        image = image[0:height-40,0:width]       
        image = cv2.resize(image, self.single_img_size[1:], interpolation=cv2.INTER_AREA)
        tensor_observation = torch.from_numpy(image)
        area = torch.from_numpy(area)
        return tensor_observation,objetive,area

    def load_img(self):
        """
        Se cargan las imagenes de MsPacman para obtener las posiciones 
        """
        self.global_img =  [('agent',cv2.imread(f"agent_{i}.png", 0)) for i in range (1,6)] 
        self.global_img += [('bonus',cv2.imread(f"bonus.png", 0))]

    def reset_episode(self, environment):
        """
        Se reinicia el episodio.
        """
        observation,objects,area = self.preprocess_observation(environment.reset())
        for i in range(self.frames_num):
            self.multi_frames[i] = observation
        multi_frames_ = self.multi_frames
        done, rewards = False, 0
        for _ in range(self.no_op_num):
            multi_frames_, step_rewards, done,_,_ = self.go_steps(multi_frames_, 0)
            rewards += step_rewards
        self.multi_frames = multi_frames_
        return done, rewards, multi_frames_,objects,area

    def update_multi_frames(self, observation_):
        """
        Se actualizan los frames acumulados.
        """
        multi_frames_ = self.multi_frames.clone().detach()
        for i in range(self.frames_num - 1):
            multi_frames_[self.frames_num - 1 - i, :] = multi_frames_[self.frames_num - 2 - i, :]
        multi_frames_[0, :] = observation_
        return multi_frames_

    def sample(self):
        """
        Se retorna la memoria.
        """
        return self.memory.sample(self.batch_size)


    def soft_update_target(self, target_model, behavior_model):
        """
        Se actualiza el target
        """
        for target_param, local_param in zip(target_model.parameters(), behavior_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_memory(self, r, action, multi_frames_, done,obj,area):
        """
        Se guarda los parametros en el buffer
        """
        if self.reward_clip:
            r = min(max(r, -self.step_num), self.step_num)

        self.memory.store_sars_(self.multi_frames.to('cpu'),
                                torch.Tensor([action]), torch.Tensor([r]), multi_frames_, torch.Tensor([done]),torch.Tensor([obj]),area)

    def go_steps(self, multi_frames_: torch.Tensor, action: int):
        """
        Se realizan los paso y se acumulan los frames
        """
        step_rewards, done = 0, None
        for _ in range(self.step_num):
            observation_, reward, done, info = self.environment.step(action)
            obs,obj,area = self.preprocess_observation(observation_)
            step_rewards += reward
            if obj:
                step_rewards+=1000
            multi_frames_ = self.update_multi_frames(obs)
        self.step_count += 1
        return multi_frames_, step_rewards, done,obj,area

    def train(self, net_path=None, start_episodes=0, eval=False, start_frames=0):
        """
        Se comienza la fase de entrenamiento
        """
        self.frames_count = start_frames
        self.load_model(net_path, eval, start_episodes)
        
        for episode in range(start_episodes, self.episodes):
            self.gen = None
            self.gen = self.aleatorio(self.opciones)
            self.area = next(self.gen)
            done, score, multi_frames_,_,_= self.reset_episode(self.environment)
            while not done:
                #self.environment.render()
                action = self.get_action(self.multi_frames, eval)
                multi_frames_, step_rewards, done,objetive,area = self.go_steps(multi_frames_, action)
                score += step_rewards
                self.frames_count += self.step_num
                if not eval:
                    self.save_memory(step_rewards, action, multi_frames_, done,objetive,area)
                    self.learn()
                    
                self.multi_frames = multi_frames_  
                 
               
            self.scores[episode] = score
            self.process_results(episode, eval)
            if episode % 10 == 9:
                self.writer.add_scalar("Reward/Episode", np.mean(self.scores[episode - 9:episode]), episode+1)    
        self.writer.flush()
        self.writer.close 

