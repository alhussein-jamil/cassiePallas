import numpy as np
import gymnasium.utils as utils 
import mujoco as m 
import gymnasium as gym 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from gymnasium.spaces  import Box
import torch
import os 
import cv2 
import ray

from ray.rllib.agents.ppo import PPOTrainer

#First we need to define the environment

#The constants are defined here
THETA_LEFT = 0.5
THETA_RIGHT = 0
MAX_STEPS = 300 
OMEGA = 1 
STEPS_IN_CYCLE= 20 
a_swing = 0 
b_swing = 0.5
a_stance = 0.5
b_stance = 1
FORWARD_QUARTERNIONS = np.array([1, 0, 0, 0])
KAPPA = 200
X_VEL = 0.2
Z_VEL = 0
c_swing_frc = -1 
c_stance_frc = 0
c_swing_spd = 0
c_stance_spd = -1

#The camera configuration
DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,  # use the body id of Cassie
    "distance": 4.0,
    "lookat": np.array((0.0, 0.0, 0.85)),  # adjust the height to match Cassie's height
    "elevation": -20.0,
}


#The environment class
class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }




    def vm_cdf(self,x, mu, kappa, num_points=1000):
        """Computes the CDF of the von Mises distribution.

        Parameters:
        x (float or array-like): Value(s) at which to evaluate the CDF.
        mu (float): Mean parameter of the von Mises distribution.
        kappa (float): Concentration parameter of the von Mises distribution.
        num_points (int, optional): Number of points to use in the numerical integration. Default is 1000.

        Returns:
        float or array-like: The CDF of the von Mises distribution evaluated at `x`.
        """
        
        def besseli0(x):
            """Approximation of the Bessel I0 function."""
            ax = np.abs(x)
            if ax < 3.75:
                y = x / 3.75
                y2 = y ** 2
                return 1.0 + y2 * (3.5156229 + y2 * (3.0899424 + y2 * (1.2067492 + y2 * (0.2659732 + y2 * (0.0360768 + y2 * 0.0045813)))))
            else:
                y = 3.75 / ax
                return (np.exp(ax) / np.sqrt(ax)) * (0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (-0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377))))))))

        def integrand(t):
            """Integrand of the CDF."""
            return np.exp(kappa * np.cos(t - mu)) / besseli0(kappa)

        if isinstance(x, (int, float)):
            # Compute the numerical integral using the trapezoidal rule
            xvals = np.linspace(-np.pi, x, num_points)
            yvals = integrand(xvals)
            integral = np.trapz(yvals, xvals)

            # Compute the normalization constant
            zvals = integrand(np.linspace(-np.pi, np.pi, num_points))
            normalization = np.trapz(zvals, np.linspace(-np.pi, np.pi, num_points))

            # Return the CDF
            return integral / normalization
        else:
            cdf_values = []
            for xi in x:
                # Compute the numerical integral using the trapezoidal rule
                xvals = np.linspace(-np.pi, xi, num_points)
                yvals = integrand(xvals)
                integral = np.trapz(yvals, xvals)

                # Compute the normalization constant
                zvals = integrand(np.linspace(-np.pi, np.pi, num_points))
                normalization = np.trapz(zvals, np.linspace(-np.pi, np.pi, num_points))

                # Append the CDF value to the list
                cdf_values.append(integral / normalization)
            return np.array(cdf_values)

    def __init__(self,config,  **kwargs):
        utils.EzPickle.__init__(self, config, **kwargs)

        self._forward_reward_weight = config.get("forward_reward_weight", 1.25)
        self._ctrl_cost_weight = config.get("ctrl_cost_weight", 0.1)
        self._healthy_reward = config.get("healthy_reward", 5.0)
        self._terminate_when_unhealthy = config.get("terminate_when_unhealthy", True)
        self._healthy_z_range = config.get("healthy_z_range", (0.5, 2.0))
        actuator_ranges = {
            'left-hip-roll': [-4.5, 4.5],
            'left-hip-yaw': [-4.5, 4.5],
            'left-hip-pitch': [-12.2, 12.2],
            'left-knee': [-12.2, 12.2],
            'left-foot': [-0.9, 0.9],
            'right-hip-roll': [-4.5, 4.5],
            'right-hip-yaw': [-4.5, 4.5],
            'right-hip-pitch': [-12.2, 12.2],
            'right-knee': [-12.2, 12.2],
            'right-foot': [-0.9, 0.9]
        }
        
        # create the action space using the actuator ranges
        low = [actuator_ranges[key][0] for key in actuator_ranges.keys()]
        high = [actuator_ranges[key][1] for key in actuator_ranges.keys()]
        self.action_space = gym.spaces.Box(np.array(low), np.array(high), dtype=np.float32)
        self._reset_noise_scale = config.get("reset_noise_scale", 1e-2)
        self.phi = 0
        self._exclude_current_positions_from_observation = config.get("exclude_current_positions_from_observation", True)
        self.steps =0
        self.previous_action = np.zeros (10)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float64)
        MujocoEnv.__init__(self, config.get("model_path","cassie.xml") ,20,render_mode=config.get("render_mode",None), observation_space=observation_space,  **kwargs)
        #set the camera settings to match the DEFAULT_CAMERA_CONFIG we defined above


    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        is_healthy = min_z < self.data.qpos[2] < max_z

        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if (self._terminate_when_unhealthy or self.steps>MAX_STEPS)  else False
        return terminated
    def _get_obs(self):
        '''The sensor data are the following 
        left-foot-input [-2.20025499]
        left-foot-output [-0.0440051]
        left-hip-pitch-input [0.03050987]
        left-hip-roll-input [0.2013339]
        left-hip-yaw-input [0.30352121]
        left-knee-input [-12.57921603]
        left-shin-output [-0.00254359]
        left-tarsus-output [1.01621498]
        pelvis-angular-velocity [-0.66091646  0.09304743 -0.14707221]
        pelvis-linear-acceleration [  0.18789681 -11.51133484  -0.64882624]
        pelvis-magnetometer [ 3.71266894e-04 -4.99997057e-01 -1.67494055e-03]
        pelvis-orientation [ 9.99998508e-01 -1.67501875e-03  2.04083784e-04 -3.70925603e-04]
        right-foot-input [-2.18139208]
        right-foot-output [-0.04362784]
        right-hip-pitch-input [0.00453772]
        right-hip-roll-input [0.12788968]
        right-hip-yaw-input [0.08293241]
        right-knee-input [-12.53914917]
        right-shin-output [-0.00094644]
        right-tarsus-output [1.01240756]    
        '''
        p =np.array ([np.sin((2*np.pi*(self.phi+THETA_LEFT))),np.sin((2*np.pi*(self.phi+THETA_RIGHT)))])

        #getting the read positions of the sensors and concatenate the lists
        return np.concatenate([self.data.sensordata,p])

    def get_pos(self):
                
        #Robot State
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        #Desired velocity


        #Phase ratios and clock inputs

        #p = {sin(2pi(phi+theta_left)/L),sin(2pi(phi+theta_right)/L)} where L is the number of timesteps in each period
        p = (np.sin((2*np.pi*(self.phi+THETA_LEFT))),np.sin((2*np.pi*(self.phi+THETA_RIGHT))))
        '''
		Position [1], [2] 				-> Pelvis y, z
				 [3], [4], [5], [6] 	-> Pelvis Orientation qw, qx, qy, qz
				 [7], [8], [9]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [14]					-> Left Knee   	(Motor[3])
				 [15]					-> Left Shin   	(Joint[0])
				 [16]					-> Left Tarsus 	(Joint[1])
				 [20]					-> Left Foot   	(Motor[4], Joint[2])
				 [21], [22], [23]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [28]					-> Rigt Knee   	(Motor[8])
				 [29]					-> Rigt Shin   	(Joint[3])
				 [30]					-> Rigt Tarsus 	(Joint[4])
				 [34]					-> Rigt Foot   	(Motor[9], Joint[5])
		''' 
        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        
        '''
		Velocity [0], [1], [2] 			-> Pelvis x, y, z
				 [3], [4], [5]		 	-> Pelvis Orientation wx, wy, wz
				 [6], [7], [8]			-> Left Hip Roll (Motor[0]), Yaw (Motor[1]), Pitch (Motor[2])
				 [12]					-> Left Knee   	(Motor[3])
				 [13]					-> Left Shin   	(Joint[0])
				 [14]					-> Left Tarsus 	(Joint[1])
				 [18]					-> Left Foot   	(Motor[4], Joint[2])
				 [19], [20], [21]		-> Right Hip Roll (Motor[5]), Yaw (Motor[6]), Pitch (Motor[7])
				 [25]					-> Rigt Knee   	(Motor[8])
				 [26]					-> Rigt Shin   	(Joint[3])
				 [27]					-> Rigt Tarsus 	(Joint[4])
				 [31]					-> Rigt Foot   	(Motor[9], Joint[5])
		''' 
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
        return np.concatenate([qpos[pos_index], qvel[vel_index],[p[0],p[1]]])
    

    def von_mises(a,kappa,phi ):
        vm = torch.distributions.von_mises(a,kappa)
        return vm.cdf(phi)
    
    #computes the reward
    def compute_reward(self,action):

        # Extract some proxies
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()
        pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])
        
        qpos = qpos[pos_index]
        qvel=qvel[vel_index]


        #Feet Contact Forces 
        contact_force_right_foot = np.zeros(6)
        m.mj_contactForce(self.model,self.data,0,contact_force_right_foot)
        contact_force_left_foot = np.zeros(6)
        m.mj_contactForce(self.model,self.data,1,contact_force_left_foot)


        # Update previous position

        ######## Odometry xy reward ########
        q_vx = 1-np.exp(-2*OMEGA*np.linalg.norm(np.array([qvel[0]]) - np.array([X_VEL]))**2)
        ################

        ######## Odometry xy reward ########
        q_vy = 1-np.exp(-2*OMEGA*np.linalg.norm(np.array([qvel[2]]) - np.array([Z_VEL]))**2)
        ################

        q_left_frc = 1.0 - np.exp(-OMEGA * np.linalg.norm(contact_force_left_foot)**2/4000)
        q_right_frc = 1.0 - np.exp(-OMEGA * np.linalg.norm(contact_force_right_foot)**2/4000)
        q_left_spd = 1.0 - np.exp(-OMEGA * np.linalg.norm(qvel[12])**2)
        q_right_spd = 1.0 - np.exp(-OMEGA * np.linalg.norm(qvel[19])**2)
        

        q_action_diff = 1 - np.exp(-5*np.linalg.norm(action-self.previous_action))
        q_orientation = 1 -np.exp(-3*(1-((self.data.sensor('pelvis-orientation').data.T)@(FORWARD_QUARTERNIONS))**2))
        q_torque = 1 - np.exp(-0.05*np.linalg.norm(action))
        q_pelvis_acc = 1 - np.exp(-0.10*(np.linalg.norm(self.data.sensor('pelvis-angular-velocity').data) + np.linalg.norm(self.data.sensor('pelvis-linear-acceleration').data)))
        

        I = lambda phi,a,b : self.vm_cdf(phi,a,KAPPA)*(1-self.vm_cdf(phi,b,KAPPA))

        I_swing_frc = lambda phi : I(phi,a_swing,b_swing)
        I_swing_spd = lambda phi : I(phi, a_swing,b_swing)
        I_stance_spd = lambda phi : I(phi, a_stance,b_stance)
        I_stance_frc = lambda phi : I(phi, a_stance,b_stance)
        C_frc = lambda phi : c_swing_frc * I_swing_frc(phi) + c_stance_frc * I_stance_frc(phi) + c_stance_frc * I_stance_frc(phi)

        C_spd = lambda phi :  c_swing_spd * I_swing_spd(phi) + c_stance_spd * I_stance_spd(phi)
        
        R_cmd = -1.0*q_vx-1.0*q_vy-1.0*q_orientation

        R_smooth = -1.0*q_action_diff - 1.0* q_torque - 1.0*q_pelvis_acc


        R_biped = 0
        R_biped += C_frc(self.phi+THETA_LEFT) * q_left_frc
        R_biped += C_frc(self.phi+THETA_RIGHT) * q_right_frc
        R_biped += C_spd(self.phi+THETA_LEFT) * q_left_spd
        R_biped += C_spd(self.phi+THETA_RIGHT) * q_right_spd

        reward = 1.5  + 0.5 * R_biped  +  0.375* R_cmd +  0.125* R_smooth
        #store all used values with their names in a dictionary
        self.rewards = {
            'R_biped': R_biped,
            'R_cmd': R_cmd,
            'R_smooth': R_smooth,
            'q_vx': q_vx,
            'q_vy': q_vy,
            'q_orientation': q_orientation,
            'q_action_diff': q_action_diff,
            'q_torque': q_torque,
            'q_pelvis_acc': q_pelvis_acc,
            'q_left_frc': q_left_frc,
            'q_right_frc': q_right_frc,
            'q_left_spd': q_left_spd,
            'q_right_spd': q_right_spd,
            'reward': reward,
            'contact_force_left_foot': np.linalg.norm(contact_force_left_foot)**2/4000
        }
        return reward
    
    #step in time
    def step(self, action):
        #clip the action to the ranges in action_space
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        self.do_simulation(action, self.frame_skip)


        observation = self._get_obs()
        
        reward = self.compute_reward(action)

        terminated = self.terminated

        self.steps +=1 
        self.phi+= 1.0/STEPS_IN_CYCLE
        self.phi = self.phi % 1 

        if self.render_mode == "human":
            self.render()
        self.previous_action = action 
        return observation, reward, terminated, False, {}

    #resets the simulation
    def reset_model(self):

        m.mj_inverse(self.model, self.data)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.previous_action = np.zeros (10)
        self.phi = 0 
        self.steps = 0 
        
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation



import os
log_dir = "/home/ajvendetta/ray_results"
sim_dir = "./sim/"
checkpoint_path = None
#load the trainer from the latest checkpoint if exists 
#get the full directory of latest modified directory in the log_dir 
if(os.path.exists(log_dir)):
    latest_log_directory = max([d for d in os.listdir(log_dir) if d.startswith("PPO_")], default=0)
    print(latest_log_directory)
    #check that the folder is not empty
    if(latest_log_directory == 0):
        print("No checkpoints found")
    else:     
        #get the latest directory in the latest log directory
        latest_directory = max([d.split("_")[-1] for d in os.listdir(os.path.join(log_dir, latest_log_directory)) if d.startswith("checkpoint")], default=0)
        #load the trainer from the latest checkpoint
        checkpoint_path = os.path.join(log_dir, latest_log_directory, "checkpoint_{}/".format(latest_directory, latest_directory))
        print(checkpoint_path)

#register the environment in rllib 

#import the necessary libraries to initialize ray and register_env

from ray.tune.registry import register_env


#initialize ray and choose the log directory



#initialize ray and register the environment
ray.init(ignore_reinit_error=True)
register_env("cassie-v0", lambda config: CassieEnv(config))


config = {
    "framework": "torch",
    "log_level": "WARN",
    "num_gpus": 1,
    "num_cpus": 20,
    "num_workers": 20,
    "num_envs_per_worker": 1,
    "rollout_fragment_length": 300,
    "train_batch_size": 50000,
    "sgd_minibatch_size": 9000,
    "num_sgd_iter": 10,
    "optimizer": {
        "type": "Adam",
        "lr": 3e-4,
        "epsilon": 1e-5
    },
    "model": {
        "conv_filters": None,
        "fcnet_activation": "swish",
        "fcnet_hiddens": [128, 128, 64],
        "vf_share_layers": False,
        "free_log_std": True
    },
    "entropy_coeff": 0.01,
    "gamma": 0.99,
    "lambda": 0.95,
    "kl_coeff": 0.5,
    "clip_param": 0.2,
    "num_workers": 6,

    "batch_mode": "truncate_episodes",
    "observation_filter": "NoFilter",
    "reuse_actors": True,
    "disable_env_checking": True,
    "num_gpus_per_worker": 0.05,
    "num_cpus_per_worker": 1,

}
import tensorflow as tf

import tensorboard

torch.cuda.empty_cache()
if(checkpoint_path is not None):
    temp = PPOTrainer(config, "cassie-v0")
    temp.restore(checkpoint_path)

    # Get policy weights
    policy_weights = temp.get_policy().get_weights()

    # Destroy temp
    temp.stop()

trainer = PPOTrainer(config=config, env="cassie-v0")
if(checkpoint_path is not None):
    # Set the policy weights to the second trainer
    trainer.get_policy().set_weights(policy_weights)


import cv2
import os

# Training loop

max_test_i = 0
checkpoint_frequency = 5
simulation_frequency = 10
env = CassieEnv({})
env.render_mode = "rgb_array"



# Find the latest directory named test_i in the sim directory
latest_directory = max([int(d.split("_")[-1]) for d in os.listdir(sim_dir) if d.startswith("test_")], default=0)
max_test_i = latest_directory + 1

# Create folder for test
test_dir = os.path.join(sim_dir, "test_{}".format(max_test_i))
os.makedirs(test_dir, exist_ok=True)





# Define video codec and framerate
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30



# Set initial iteration count
i = trainer.iteration if hasattr(trainer, "iteration") else 0
#training 
print("Starting training loop")
while True:

        # Train for one iteration
        result = trainer.train()
        i += 1
        print("Episode Reward Mean for iteration {} is {}".format(i, result["episode_reward_mean"]))

        # Save model every 10 epochs
        if i % checkpoint_frequency == 0:
            checkpoint_path = trainer.save()
            print("Checkpoint saved at", checkpoint_path)

        # Run a test every 20 epochs
        if i % simulation_frequency == 0:
            #make a steps counter
            steps = 0

            # Run test
            video_path = os.path.join(test_dir, "sim_{}.mp4".format(i))

            env.reset()
            obs = env.reset()[0]
            done = False
            frames = []

            while not done:

                # Increment steps
                steps += 1
                action = trainer.compute_single_action(obs)
                obs, _, done, _, _ = env.step(action)
                frame = env.render()

                #show the step number on the frame
                cv2.putText(frame, "Step: {}".format(steps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames.append(frame)

            # Save frames as video
            height, width, _ = frames[0].shape
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
            for frame in frames:
                video_writer.write(frame)
            video_writer.release()

            # Increment test index
            max_test_i += 1

