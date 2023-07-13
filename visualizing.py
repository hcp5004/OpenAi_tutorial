import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from collections import defaultdict
import tilecoding

class visualizing:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.show()
    
    def show(self):
        rolling_length = 1
        fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
        axs[0].set_title("Episode rewards")
        # compute and assign a rolling average of the data to provide a smoother graph
        reward_moving_average = (
            np.convolve(
                np.array(self.env.return_queue).flatten(), np.ones(rolling_length), mode="valid"
            )
            / rolling_length
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[1].set_title("Episode lengths")
        length_moving_average = (
            np.convolve(
                np.array(self.env.length_queue).flatten(), np.ones(rolling_length), mode="same"
            )
            / rolling_length
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[2].set_title("Training Error")
        training_error_moving_average = (
            np.convolve(np.array(self.agent.training_error).flatten(), np.ones(rolling_length), mode="same")
            / rolling_length
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        plt.tight_layout()
        # plt.show()

class policy:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.create_grids()
        self.create_plots(title= "State")
        #self.show()

    def create_grids(self):
        """Create value and policy grid given an agent."""
        # convert our state-action values to state values
        # and build a policy dictionary that maps observations to actions
        state_value = defaultdict(float)
        #policy = defaultdict(float)
        # for obs, action_values in self.agent.q_values.items():
        #     state_value[obs] = float(np.max(action_values))
        #     policy[obs] = int(np.argmax(action_values))

        for x_n in range(0,64):
            for xdot_n in range(0,64):
                x = x_n*(0.6+1.2)/64 - 1.2
                xdot = xdot_n*(0.07 + 0.07)/64 - 0.07

                obs_x_1 = np.zeros(self.agent.w.shape) 
                obs_x_1[tilecoding.tiles(self.agent.iht, 8, [8*x/(0.6+1.2), 8*xdot/(0.07 + 0.07)] , [2])] = 1

                obs_x_2 = np.zeros(self.agent.w.shape) 
                obs_x_2[tilecoding.tiles(self.agent.iht, 8, [8*x/(0.6+1.2), 8*xdot/(0.07 + 0.07)] , [0])] = 1

                q_value_1 = self.agent.w.T@obs_x_1
                q_value_2 = self.agent.w.T@obs_x_2
                state_value[(x,xdot)] = float(q_value_1 if q_value_1> q_value_2 else q_value_2)


        position, velocity = np.meshgrid(
            np.array([x_n*(0.6+1.2)/64 - 1.2 for x_n in range(0, 64)]),
            np.array([xdot_n*(0.07 + 0.07)/64 - 0.07 for xdot_n in range(0,64)]),
        )

        # create the value grid for plotting
        value = np.apply_along_axis(
            lambda obs: state_value[(obs[0], obs[1])],
            axis=2,
            arr=np.dstack([position, velocity]),
        )
        self.value_grid = position, velocity, value

    def create_plots(self, title: str, policy_grid = None):
        """Creates a plot using a value and policy grid."""
        # create a new figure with 2 subplots (left: state values, right: policy)
        position, velocity, value = self.value_grid
        fig = plt.figure(figsize=plt.figaspect(0.4))
        fig.suptitle(title, fontsize=16)

        # plot the state values
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax1.plot_surface(
            position,
            velocity,
            value,
            rstride=1,
            cstride=1,
            cmap="viridis",
            edgecolor="none",
        )
        # plt.xticks( [x_n*(0.5+1.2)/64 - 0.5 for x_n in range(0, 64)], [x_n*(0.5+1.2)/64 - 0.5 for x_n in range(0, 64)])
        # plt.yticks( [xdot_n*(0.07 + 0.07)/64 - 0.07 for xdot_n in range(0,64)],[xdot_n*(0.07 + 0.07)/64 - 0.07 for xdot_n in range(0,64)]   )
        ax1.set_title(f"State values: {title}")
        ax1.set_xlabel("Position")
        ax1.set_ylabel("Velocity")
        ax1.zaxis.set_rotate_label(False)
        ax1.set_zlabel("Value", fontsize=14, rotation=90)
        ax1.view_init(20, 220)
        
        self.fig = fig

    def show(self):
        plt.show()