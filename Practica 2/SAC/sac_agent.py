import torch
import numpy as np

from torch import optim
from sac_net import QNet, PolicyNetwork
from agent import AtariAgent



class SAC(AtariAgent):
    def __init__(self, action_space, memory_par, game, reward_clip = False):
        """
        Args:
            action_space: Contiene las acciones del entorno a utilizar.
            memory_par: Contiene el tamaño del espacio de memoria y el tamaño de la imagen del entorno.
            game: Contiene el entorno.
            reward_clip: Recompensa normalizada.
        """
        super(SAC, self).__init__(action_space, memory_par, game, reward_clip)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.soft = True
        self.learn_replace = 1000
        self.learn_start_step = 1000
        self.tau = 0.005
        self.step_num = 4
        self.learn_interval = 4
        self.actor = PolicyNetwork(img_size=self.multi_frames_size, output_num=self.action_space_len)
        self.critic1 = QNet(img_size=self.multi_frames_size, output_num=self.action_space_len)
        self.critic2 = QNet(img_size=self.multi_frames_size, output_num=self.action_space_len)
        with torch.no_grad():
            self.critic1_target = QNet(img_size=self.multi_frames_size, output_num=self.action_space_len)
            self.critic2_target = QNet(img_size=self.multi_frames_size, output_num=self.action_space_len)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.target_entropy = 0.98 * (-np.log(1 / self.action_space_len))
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()

        self.critic1_optimiser = optim.Adam(self.critic1.parameters(), lr=0.0001, eps=1e-6)
        self.critic2_optimiser = optim.Adam(self.critic2.parameters(), lr=0.0001, eps=1e-6)
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=0.0001, eps=1e-6)
        self.alpha_optimiser = optim.Adam([self.log_alpha], lr=0.0001, eps=1e-6)

    def cal_critic_loss(self, s, a, r, t, s_):
        """
        Se calcula la perdida de los criticos.
        """
        with torch.no_grad():
            _, a_probabilities, log_a_probabilities = \
                self.actor.sample_action(s_.to(self.device).float())
            target_q1 = self.critic1_target.forward(s_.to(self.device).float())
            target_q2 = self.critic2_target.forward(s_.to(self.device).float())
            q_target = (a_probabilities * (torch.min(target_q1, target_q2) - self.alpha * log_a_probabilities)).sum(
                dim=1)
            terminal = (~t.to(self.device))
            q_ = r.to(self.device).float() + self.gamma * terminal * q_target

        q1 = self.critic1(s.to(self.device).float())[np.arange(self.batch_size), np.array(a)]
        q2 = self.critic2(s.to(self.device).float())[np.arange(self.batch_size), np.array(a)]
        q1_loss = torch.nn.functional.smooth_l1_loss(q1, q_)
        q2_loss = torch.nn.functional.smooth_l1_loss(q2, q_)
        return q1_loss, q2_loss

    def update_critics(self, s, a, r, t, s_):
        """
        Se actualizan los valores de los criticos.
        """
        q1_loss, q2_loss = self.cal_critic_loss(s, a, r, t, s_)

        self.critic1_optimiser.zero_grad()
        q1_loss.backward()
        self.critic1_optimiser.step()

        self.critic2_optimiser.zero_grad()
        q2_loss.backward()
        self.critic2_optimiser.step()

    def cal_actor_alpha_loss(self, s):
        """
        Calcula la perdida del actor
        """
        _, a_probabilities, log_a_probabilities = self.actor.sample_action(s.to(self.device).float())
        with torch.no_grad():
            q1 = self.critic1(s.to(self.device).float())
            q2 = self.critic2(s.to(self.device).float())
        q = torch.min(q1, q2)

        actor_loss = (a_probabilities * (self.alpha * log_a_probabilities - q)).sum(dim=1).mean()
        probabilities = (a_probabilities * log_a_probabilities).sum(dim=1)
        alpha_loss = -(self.log_alpha * (probabilities + self.target_entropy).detach()).mean()
        return actor_loss, alpha_loss

    def update_actor_alpha(self, s):
        """
        Actualiza el actor
        """
        actor_loss, alpha_loss = self.cal_actor_alpha_loss(s)

        self.actor_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_optimiser.step()

        self.alpha_optimiser.zero_grad()
        alpha_loss.backward()
        self.alpha_optimiser.step()
        self.alpha = self.log_alpha.exp()

    def update_target(self):
        """
        Actualiza el target
        """
        if self.soft:
            self.soft_update_target(self.critic1_target, self.critic1)
            self.soft_update_target(self.critic2_target, self.critic2)
        else:
            if self.learn_cur % self.learn_replace == 0:
                self.critic1_target.load_state_dict(self.critic1.state_dict())
                self.critic2_target.load_state_dict(self.critic2.state_dict())

    def learn(self):
        """
        Se aprende de la memoria cada x pasos
        """
        if self.step_count < self.learn_start_step or self.step_count % self.learn_interval != 0:
            return

        s, a, r, s_, t = self.sample()
        self.update_critics(s, a, r, t, s_)
        self.update_actor_alpha(s)
        self.update_target()
        self.learn_cur += 1

    def get_action(self, s, eval=False):
        """
        Si obtienen las acciones.
        """
        if eval:
            with torch.no_grad():
                action = self.actor.get_best_action(s[None, ...].to(self.device))
        else:
            if self.step_count < 20000:
                action = self.action_space[np.random.randint(self.action_space_len)]
            else:
                with torch.no_grad():
                    action, _, _ = self.actor.sample_action(s[None, ...].to(self.device))
                    action = action.item()
        return action

    def process_results(self, episode, eval):
        if episode % 10 == 9:
            ave = np.mean(self.scores[episode - 9:episode])
            print('Episodes: {}, AveScores: {}, Steps: {}'.format(
                episode + 1, ave,self.step_count))
        if eval:
            if episode % 10 == 99:
                s1 = './' + self.game_name + '/'
                np.save(s1 + 'scores_eval{}.npy'.format(episode + 1), self.scores)
                print('Evaluation results saved!')
        else:
            if episode % 10000 == 199:
                self.save_episode_models(episode)
                print('Model salved!')
                print('Total {} frames!'.format(self.frames_count))

    def save_episode_models(self, episode):
        """
        Se guarda el modelo.
        """
        s1 = 'SAC/runs/' + self.game_name + '/'
        s_pth = '{}.pth'.format(episode + 1)
        torch.save(self.actor.state_dict(), s1 + 'actor' + s_pth)
        torch.save(self.critic1.state_dict(), s1 + 'critic1_' + s_pth)
        torch.save(self.critic2.state_dict(), s1 + 'critic2_' + s_pth)
        torch.save(self.critic1_target.state_dict(), s1 + 'critic1_target' + s_pth)
        torch.save(self.critic2_target.state_dict(), s1 + 'critic2_target' + s_pth)

        torch.save(self.log_alpha, s1 + 'log_alpha' + s_pth)

        torch.save(self.actor_optimiser.state_dict(), s1 + 'actor_optimizer' + s_pth)
        torch.save(self.critic1_optimiser.state_dict(), s1 + 'critic1_optimizer' + s_pth)
        torch.save(self.critic2_optimiser.state_dict(), s1 + 'critic2_optimizer' + s_pth)
        torch.save(self.alpha_optimiser.state_dict(), s1 + 'alpha_optimizer' + s_pth)
        np.save(s1 + 'scores{}.npy'.format(episode + 1), self.scores)

    def load_model(self, net_path, eval=False, start_episodes=0):
        """
        Se carga el modelo.
        """
        if eval:
            self.actor.load_state_dict(torch.load(net_path + '/actor.pth'))
            self.actor.eval()
        if start_episodes != 0 and not eval:
            self.actor.load_state_dict(torch.load(net_path + '/actor{}.pth'.format(start_episodes)))
            s1 = net_path + '/'
            s2 = '{}.pth'.format(start_episodes)
            self.critic1.load_state_dict(torch.load(s1 + 'critic1_' + s2))
            self.critic2.load_state_dict(torch.load(s1 + 'critic2_' + s2))
            self.critic1_target.load_state_dict(torch.load(s1 + 'critic1_target' + s2))
            self.critic2_target.load_state_dict(torch.load(s1 + 'critic2_target' + s2))
            self.log_alpha = torch.load(s1 + 'log_alpha' + s2, map_location=self.device)

            self.actor_optimiser.load_state_dict(torch.load(s1 + 'actor_optimizer' + s2))
            self.critic1_optimiser.load_state_dict(torch.load(s1 + 'critic1_optimizer' + s2))
            self.critic2_optimiser.load_state_dict(torch.load(s1 + 'critic2_optimizer' + s2))
            self.alpha_optimiser.load_state_dict(torch.load(s1 + 'alpha_optimizer' + s2))

            self.scores = np.load(net_path + '/scores{}.npy'.format(start_episodes))
            self.learn_cur += 1
