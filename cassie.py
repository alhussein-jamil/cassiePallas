import constants as c 
import gymnasium.utils as utils 
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
import functions as f 
import numpy as np
import gymnasium as gym 
from gymnasium.spaces import Box
import mujoco as m 
import torch



class CassieEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 100,
    }


    def __init__(self,config,  **kwargs):
        utils.EzPickle.__init__(self, config, **kwargs)
        self._terminate_when_unhealthy = config.get("terminate_when_unhealthy", True)
        self._healthy_z_range = config.get("healthy_z_range", (0.35, 2.0))
 
        # create the action space using the actuator ranges
        low = [c.actuator_ranges[key][0] for key in c.actuator_ranges.keys()]
        high = [c.actuator_ranges[key][1] for key in c.actuator_ranges.keys()]
        self.action_space = gym.spaces.Box(np.float32(np.array(low)), np.float32(np.array(high)))
        self._reset_noise_scale = config.get("reset_noise_scale", 1e-2)
        self.phi = 0
        self.steps =0
        self.previous_action = torch.zeros(10)
        low = [-3]*23
        low.append(-1)
        low.append(-1)
        high = [3]*23
        high.append(1)
        high.append(1)
        self.gamma = config.get("gamma",0.99)
        self.gamma_modified = 1
        self.rewards = {"R_biped":0,"R_cmd":0,"R_smooth":0}
        self.observation_space = Box(low=np.float32(np.array(low)), high=np.float32(np.array(high)), shape=(25,))

        MujocoEnv.__init__(self, config.get("model_path","/home/alhussein.jamil/Cassie/cassie-mujoco-sim-master/model/cassie.xml"),20 ,render_mode=config.get("render_mode",None), observation_space=self.observation_space,  **kwargs)
        #set the camera settings to match the DEFAULT_CAMERA_CONFIG we defined above



    
    @property
    def healthy_reward(self):
        return float(self.is_healthy or self._terminate_when_unhealthy) * self._healthy_reward

    @property
    def is_healthy(self):
        min_z, max_z = self._healthy_z_range
        #it is healthy if in range and one of the feet is on the ground
        is_healthy = min_z < self.data.qpos[2] < max_z 
        return is_healthy

    @property
    def terminated(self):
        terminated = (not self.is_healthy) if (self._terminate_when_unhealthy or self.steps>c.MAX_STEPS)  else False
        return terminated
    def _get_obs(self):


        p =np.array ([np.sin((2*np.pi*(self.phi))),np.cos((2*np.pi*(self.phi)))])
        temp = []
        #normalize the sensor data using sensor_ranges self.data.sensor('pelvis-orientation').data
        for key in c.sensor_ranges.keys():
            temp.append(f.normalize(key,self.data.sensor(key).data))

        temp = np.array(np.concatenate(temp))

        #getting the read positions of the sensors and concatenate the lists
        return np.concatenate([temp,p])

    def get_pos(self):
                
        #Robot State in simulator
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
        qpos = self.data.qpos.flat.copy()

            
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
        qvel = self.data.qvel.flat.copy()

        return np.concatenate([qpos[c.pos_index], qvel[c.vel_index]])
    
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


        #Some metrics to be used in the reward function
        q_vx = 1-np.exp(-2*c.OMEGA*np.linalg.norm(np.array([qvel[0]]) - np.array([c.X_VEL]))**2)
        q_vy = 1-np.exp(-2*c.OMEGA*np.linalg.norm(np.array([qvel[1]]) - np.array([c.Y_VEL]))**2)
        q_vz = 1-np.exp(-2*c.OMEGA*np.linalg.norm(np.array([qvel[2]]) - np.array([c.Z_VEL]))**2)

        q_left_frc = 1.0 - np.exp(-c.OMEGA * np.linalg.norm(contact_force_left_foot)**2/c.q_frc_coef)
        q_right_frc = 1.0 - np.exp(-c.OMEGA * np.linalg.norm(contact_force_right_foot)**2/c.q_frc_coef)
        q_left_spd = 1.0 - np.exp(-np.linalg.norm(qvel[12])**2)
        q_right_spd = 1.0 - np.exp(-np.linalg.norm(qvel[19])**2)
        q_action_diff = 1 - np.exp(-float(f.action_dist(torch.tensor(action).reshape(1,-1),torch.tensor(self.previous_action).reshape(1,-1))))
        q_orientation = 1 -np.exp(-3*(1-((self.data.sensor('pelvis-orientation').data.T)@(c.FORWARD_QUARTERNIONS))**2))
        q_torque = 1 - np.exp(-0.05*np.linalg.norm(action))
        q_pelvis_acc = 1 - np.exp(-0.10*(np.linalg.norm(self.data.sensor('pelvis-angular-velocity').data) ))#+ np.linalg.norm(self.data.sensor('pelvis-linear-acceleration').data-self.model.opt.gravity.data)))

        #Responsable for the swing and stance phase
        I = lambda phi,a,b : f.p_between_von_mises(a,b,c.KAPPA,phi)

        I_swing_frc = lambda phi : I(phi,c.a_swing,c.b_swing)
        I_swing_spd = lambda phi : I(phi, c.a_swing,c.b_swing)
        I_stance_spd = lambda phi : I(phi, c.a_stance,c.b_stance)
        I_stance_frc = lambda phi : I(phi, c.a_stance,c.b_stance)

        C_frc = lambda phi : c.c_swing_frc * I_swing_frc(phi) + c.c_stance_frc * I_stance_frc(phi) 
        C_spd = lambda phi :  c.c_swing_spd * I_swing_spd(phi) + c.c_stance_spd * I_stance_spd(phi)
        

        R_cmd = - 1.0*q_vx - 1.0*q_vy - 1.0*q_orientation - 0.5*q_vz
        R_smooth = -1.0*q_action_diff - 1.0* q_torque - 1.0*q_pelvis_acc
        R_biped = 0
        R_biped += C_frc(self.phi+c.THETA_LEFT) * q_left_frc
        R_biped += C_frc(self.phi+c.THETA_RIGHT) * q_right_frc
        R_biped += C_spd(self.phi+c.THETA_LEFT) * q_left_spd
        R_biped += C_spd(self.phi+c.THETA_RIGHT) * q_right_spd



        reward = 4 + 0.5 * R_biped  +  0.375* R_cmd +  0.125* R_smooth
        
        self.used_quantities = {"C_frc_left":C_frc(self.phi+c.THETA_LEFT),"C_frc_right":C_frc(self.phi+c.THETA_RIGHT),"C_spd_left":C_spd(self.phi+c.THETA_LEFT),"C_spd_right":C_spd(self.phi+c.THETA_RIGHT),'q_vx':q_vx,'q_vy':q_vy,'q_vz':q_vz,'q_left_frc':q_left_frc,'q_right_frc':q_right_frc,'q_left_spd':q_left_spd,'q_right_spd':q_right_spd,'q_action_diff':q_action_diff,'q_orientation':q_orientation,'q_torque':q_torque,'q_pelvis_acc':q_pelvis_acc,'R_cmd':R_cmd,'R_smooth':R_smooth,'R_biped':R_biped}

        self.rewards['R_cmd']+=self.gamma_modified*R_cmd
        self.rewards['R_smooth']+=self.gamma_modified*R_smooth
        self.rewards['R_biped']+=self.gamma_modified*R_biped

        return reward
    
    #step in time
    def step(self, action):
        #clip the action to the ranges in action_space (done inside the config that's why removed)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        
        reward = self.compute_reward(action)

        terminated = self.terminated

        self.steps +=1 
        self.phi+= 1.0/c.STEPS_IN_CYCLE
        self.phi = self.phi % 1 

        self.previous_action = action 

        self.gamma_modified *= self.gamma
        
        return observation, reward, terminated, False, {}

    #resets the simulation
    def reset_model(self):
        m.mj_inverse(self.model, self.data)
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale
        self.previous_action = np.zeros (10)
        self.phi = 0 
        self.steps = 0 
        self.rewards = {"R_biped":0,"R_cmd":0,"R_smooth":0}

        self.gamma_modified = 1
        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nv)
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
