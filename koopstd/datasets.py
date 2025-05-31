import numpy as np
import torch
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

rng = np.random.default_rng(2023)

class Lorenz63:
    """
    A dataset class for generating Lorenz system trajectories with different rho values.
    
    This class simulates the Lorenz system for multiple rho values and generates
    trajectory clips for each value.
    
    Attributes:
        rho_values (list): List of rho parameter values to simulate.
        initial_state (tuple): Initial state (x, y, z) for the Lorenz system.
        t_start (float): Start time for simulation.
        t_end (float): End time for simulation.
        dt (float): Time step for numerical integration.
        period_length (float): Length of each data clip in time units.
        num_clips (int): Number of clips to extract from each simulation.
        data (list): Generated trajectory data.
    """
    def __init__(self, rho_values, initial_state=(-8, 8, 27), t_start=0, t_end=800, 
                 dt=0.001, period_length=20, num_clips=25, random_seed=None):
        """
        Initialize the Lorenz dataset with specified parameters.
        
        Args:
            rho_values (list): List of rho parameter values to simulate.
            initial_state (tuple): Initial state (x, y, z) for the Lorenz system.
            t_start (float): Start time for simulation.
            t_end (float): End time for simulation.
            dt (float): Time step for numerical integration.
            period_length (float): Length of each data clip in time units.
            num_clips (int): Number of clips to extract from each simulation.
            random_seed (int, optional): Seed for random number generators.
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)
        
        self.rho_values = rho_values
        self.initial_state = initial_state
        self.t_start = t_start
        self.t_end = t_end
        self.dt = dt
        self.period_length = period_length
        self.num_clips = num_clips
        self.data = self.generate_data()

    def lorenz(self, state, t, sigma=10, beta=8/3):
        """
        Lorenz system differential equations.
        
        Args:
            state (list): Current state [x, y, z].
            t (float): Current time (not used but required by odeint).
            sigma (float): Sigma parameter of the Lorenz system.
            beta (float): Beta parameter of the Lorenz system.
            
        Returns:
            list: Derivatives [dx/dt, dy/dt, dz/dt].
        """
        x, y, z = state
        dxdt = sigma * (y - x)
        dydt = x * (self.rho - z) - y
        dzdt = x * y - beta * z
        return [dxdt, dydt, dzdt]

    def generate_data(self):
        """
        Generate trajectory data for all specified rho values.
        
        Returns:
            list: List of trajectory clips, each with shape (period_length/dt, 3).
        """
        data_list = []
        t_span = (self.t_start, self.t_end)
        t_eval = np.arange(self.t_start, self.t_end, self.dt)
        
        # Calculate the index for starting valid data (after transient period)
        valid_start_idx = int(300 / self.dt)
        
        # Store valid data for each rho value for later visualization
        self.valid_data_by_rho = {}

        for rho in self.rho_values:
            self.rho = rho
            
            # Define a wrapper function to fix the unpacking issue
            def lorenz_wrapper(t, state):
                return self.lorenz(state, t)
            
            # Solve the Lorenz system using solve_ivp
            solution = solve_ivp(
                lorenz_wrapper, 
                t_span, 
                self.initial_state, 
                method='RK45', 
                t_eval=t_eval
            )
            
            # Extract solution data
            solution_data = np.vstack([solution.y[0], solution.y[1], solution.y[2]]).T
            
            # Collect data after transient period
            valid_data = solution_data[valid_start_idx:]
            valid_length = len(valid_data)
            
            # Store the valid data for this rho value
            self.valid_data_by_rho[rho] = valid_data
            
            # Calculate clip size
            clip_size = int(self.period_length / self.dt)
            
            for k in range(self.num_clips):
                # Randomly select clips from the valid data
                max_start_index = valid_length - clip_size
                if max_start_index > 0:
                    start_index = np.random.randint(0, max_start_index)
                    end_index = start_index + clip_size
                    period_data = valid_data[start_index:end_index]
                    data_list.append(period_data)
        
        return data_list
    
    def visualize_data(self, time_range=None):
        """
        Visualize the generated data with trajectories colored by rho values.
        
        Creates a 3D plot for each rho value showing the Lorenz attractor trajectories.
        
        Parameters:
        -----------
        time_range : tuple, optional
            A tuple of (start_time, end_time) to plot only a specific time range of the trajectories.
            If None, the entire trajectory is plotted.
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        
        # Determine number of unique rho values
        n_rho = len(self.rho_values)
        
        # Create figure with subplots
        fig = plt.figure(figsize=(4*n_rho, 4))
        
        # Create a subplot for each rho value
        for i, rho in enumerate(self.rho_values):
            ax = fig.add_subplot(1, n_rho, i+1, projection='3d')
            ax.set_title(f'ρ = {rho}')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Get the valid data for this rho value
            valid_data = self.valid_data_by_rho[rho]
            
            # Apply time range selection if specified
            if time_range is not None:
                start_idx = int((time_range[0] - 300) / self.dt)  # Adjust for transient period
                end_idx = int((time_range[1] - 300) / self.dt)
                # Ensure indices are within bounds
                start_idx = max(0, start_idx)
                end_idx = min(len(valid_data), end_idx)
                plot_data = valid_data[start_idx:end_idx]
            else:
                plot_data = valid_data
            
            # Plot the trajectory
            ax.plot(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2], linewidth=0.8)
        
        plt.tight_layout()
        plt.show()

def bistable_switch(cohs,params,time=100,cond_avgs=None,seeded=False):
    '''
    simple saddle model

    Parameters
    __________

    cohs : list or np.array
        list of the constant input drives (using -.1 and 0.1 as default)
    params : dict
        dictionary of the parameters in the model (a,b,c) and system evolution
        (dt, euler timesep size), (ntrials, number of sample trials),
        (sigma, noise variance)
    time : int 
        number of 'units to run', time / dt is the number of steps
    cond_avgs : np.ndarray or None
        condition average drive to control model
    seeded : bool
        if True, samples noise from a random generator with fixed seed
        
    ''' 
    a = params.get('a',-.6)
    b = params.get('b',2)
    c = params.get('c',-1)
    dt = params.get('dt',1)
    ntrials = params.get('ntrials',10)
    sigma = params.get('sigma',0)
    steps = int(time / dt)
    x = np.zeros((len(cohs),ntrials,steps,2))
    cohs = np.array(cohs)
    cohs = np.repeat(cohs[:,np.newaxis],ntrials,axis=1)
    input_optimized = np.zeros((len(cohs),ntrials,steps,2))
    for i in range(1,steps):
        dx = a*x[:,:,i-1,0]**3 + b*x[:,:,i-1,0] + cohs
        dy = c*x[:,:,i-1,1] + cohs
        dx = np.concatenate([dx[:,:,np.newaxis],dy[:,:,np.newaxis]],axis=2)
        if cond_avgs is not None:
            xavg = x[:,:,i-1].mean(axis=1)
            
            dx_avg = a*xavg[:,-1]**3 + b*xavg[:,-1] + cohs.mean(axis=1)
            dy_avg = c*xavg[:,-1] + cohs.mean(axis=1)
            dx_avg = np.concatenate([dx_avg[:,np.newaxis],dy_avg[:,np.newaxis]],axis=1)
            inp = (1/dt) * (cond_avgs[:,i] - xavg - dx_avg * dt)
            input_optimized[:,:,i] = np.repeat(inp[:,np.newaxis],ntrials,axis=1)
        else:
            input_optimized[:,:,i] = 0
        
        if seeded:
            rand = rng.normal(0,sigma,size=(len(cohs),ntrials,2))
        else:
            rand =  np.random.normal(0,sigma,size=(len(cohs),ntrials,2))
        x[:,:,i] = x[:,:,i-1] + dt * (dx + input_optimized[:,:,i] + rand)
               
    #xdot = [ax^3 + bx, cy]
    cond_avg = np.mean(x,axis=1)
    return x,cond_avg


class PDMAttractors:
    """
    A class for generating different types of attractor dynamics: bistable switch, line attractor, and point attractor.
    This class provides methods to generate data from different dynamical systems and visualize their behavior.
    Adapted from https://github.com/mitchellostrow/DSA.
    """
    
    def __init__(self, n_samples=5, n_trials=10, sigma=0, simul_step=100, dt=1, random_seed=None):
        """
        Initialize the PDMAttractors class with specified parameters.
        
        Args:
            n_samples (int): Number of parameter samples to generate.
            n_trials (int): Number of trials for each parameter set.
            sigma (float): Noise standard deviation.
            simul_step (int): Number of time units to simulate.
            dt (float): Time step for numerical integration.
            random_seed (int, optional): Seed for random number generation.
        """
        self.n_samples = n_samples
        self.n_trials = n_trials
        self.sigma = sigma
        self.simul_step = simul_step
        self.dt = dt
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def bistable_switch(self, cohs, params, time=100, cond_avgs=None, seeded=False):
        """
        Simple saddle model.
        
        Parameters:
            cohs (list or np.array): List of constant input drives (using -.1 and 0.1 as default).
            params (dict): Dictionary of model parameters (a,b,c) and system evolution parameters.
            time (int): Number of time units to run.
            cond_avgs (np.ndarray or None): Condition average drive to control model.
            seeded (bool): If True, samples noise from a random generator with fixed seed.
            
        Returns:
            tuple: (x, cond_avg) - trajectories and condition averages.
        """
        a = params.get('a', -.6)
        b = params.get('b', 2)
        c = params.get('c', -1)
        dt = params.get('dt', 1)
        ntrials = params.get('ntrials', 10)
        sigma = params.get('sigma', 0)
        steps = int(time / dt)
        
        x = np.zeros((len(cohs), ntrials, steps, 2))
        cohs = np.array(cohs)
        cohs = np.repeat(cohs[:, np.newaxis], ntrials, axis=1)
        input_optimized = np.zeros((len(cohs), ntrials, steps, 2))
        
        for i in range(1, steps):
            dx = a*x[:,:,i-1,0]**3 + b*x[:,:,i-1,0] + cohs
            dy = c*x[:,:,i-1,1] + cohs
            dx = np.concatenate([dx[:,:,np.newaxis], dy[:,:,np.newaxis]], axis=2)
            
            if cond_avgs is not None:
                xavg = x[:,:,i-1].mean(axis=1)
                
                dx_avg = a*xavg[:,-1]**3 + b*xavg[:,-1] + cohs.mean(axis=1)
                dy_avg = c*xavg[:,-1] + cohs.mean(axis=1)
                dx_avg = np.concatenate([dx_avg[:,np.newaxis], dy_avg[:,np.newaxis]], axis=1)
                inp = (1/dt) * (cond_avgs[:,i] - xavg - dx_avg * dt)
                input_optimized[:,:,i] = np.repeat(inp[:,np.newaxis], ntrials, axis=1)
            else:
                input_optimized[:,:,i] = 0
            
            if seeded:
                rand = rng.normal(0, sigma, size=(len(cohs), ntrials, 2))
            else:
                rand = np.random.normal(0, sigma, size=(len(cohs), ntrials, 2))
            
            x[:,:,i] = x[:,:,i-1] + dt * (dx + input_optimized[:,:,i] + rand)
                   
        cond_avg = np.mean(x, axis=1)
        return x, cond_avg

    def line_attractor(self, cond_avgs, params, time=100):
        """
        Line attractor model.
        
        Parameters:
            cond_avgs (np.ndarray): Condition average drive.
            params (dict): Dictionary of model parameters.
            time (int): Number of time units to run.
            
        Returns:
            tuple: (x, input_optimized) - trajectories and optimized inputs.
        """
        l0 = params.get('l0', [1, 1])
        l0 /= np.linalg.norm(l0)
        r0 = params.get('r0', [1, 0])
        sigma = params.get('sigma', 0)
        dt = params.get('dt', 1)
        ntrials = params.get('ntrials', 10)
        eval1 = params.get('eval1', -1)
        
        evals = np.diag([0, eval1])
        r1 = l0
        R = np.array([r0, r1])
        L = np.linalg.inv(R)
        A = R @ evals @ L
        theta = np.radians(45)
        c, s = np.cos(theta), np.sin(theta)
        Mrot = np.array(((c, -s), (s, c)))
        A = Mrot @ A

        steps = int(time / dt)
        x = np.zeros((cond_avgs.shape[0], ntrials, steps, 2))
        input_optimized = np.zeros((cond_avgs.shape[0], ntrials, steps, 2))
        
        for i in range(1, steps):
            dx = np.einsum('ij,mkj->mki', A, x[:,:,i-1])
            xavg = x[:,:,i-1].mean(axis=1)
            dx_avg = A @ xavg
            inp = (1/dt) * (cond_avgs[:,i] - xavg - dx_avg * dt)
            input_optimized[:,:,i] = np.repeat(inp[:,np.newaxis], ntrials, axis=1)
            x[:,:,i] = x[:,:,i-1] + dt*(dx + input_optimized[:,:,i] + 
                    np.random.normal(0, sigma, size=(cond_avgs.shape[0], ntrials, 2)))
                    
        return x, input_optimized

    def point_attractor(self, cond_avgs, params, time=100):
        """
        Point attractor model.
        
        Parameters:
            cond_avgs (np.ndarray): Condition average drive.
            params (dict): Dictionary of model parameters.
            time (int): Number of time units to run.
            
        Returns:
            tuple: (x, input_optimized) - trajectories and optimized inputs.
        """
        a1 = params.get('a1', -0.5)  # eigenvalue 1
        a2 = params.get('a2', -1)    # eigenvalue 2
        A = np.diag([-np.abs(a1), -np.abs(a2)])  # make sure they're negative
        sigma = params.get('sigma', 0)
        dt = params.get('dt', 1)
        steps = int(time / dt)
        ntrials = params.get('ntrials', 10)
        
        x = np.zeros((cond_avgs.shape[0], ntrials, steps, 2))
        input_optimized = np.zeros((cond_avgs.shape[0], ntrials, steps, 2))
        
        for i in range(1, steps):
            dx = np.einsum('ij,mkj->mki', A, x[:,:,i-1])
            xavg = x[:,:,i-1].mean(axis=1)
            dx_avg = A @ xavg
            inp = (1/dt) * (cond_avgs[:,i] - xavg - dx_avg * dt)
            input_optimized[:,:,i] = np.repeat(inp[:,np.newaxis], ntrials, axis=1)
            x[:,:,i] = x[:,:,i-1] + dt*(dx + input_optimized[:,:,i] + 
                   np.random.normal(0, sigma, size=(cond_avgs.shape[0], ntrials, 2)))
                   
        return x, input_optimized

    def get_data(self):
        """
        Generate data from all three attractor types with varying parameters.
        
        Returns:
            list: Flattened data from all attractors.
        """
        data = [[], [], []]
        # bistable args
        a = np.random.uniform(-5, -4, size=self.n_samples)  
        b = np.random.uniform(4, 5, size=self.n_samples)
        c = np.random.uniform(-4, -3, size=self.n_samples)
        # line attractor args
        eval1 = np.random.uniform(-2, -3, size=self.n_samples)
        # point attractor args
        a1 = np.random.uniform(-4, -5, size=self.n_samples)
        a2 = np.random.uniform(-8, -9, size=self.n_samples)  

        for i in range(self.n_samples):
            x1, cond_avg = self.bistable_switch(
                [-.1, .1], 
                dict(dt=self.dt, sigma=self.sigma, ntrials=self.n_trials, a=a[i], b=b[i], c=c[i]), 
                time=self.simul_step
            )  
            data[0].append(self.flatten_cond_trail(x1))

            x2, _ = self.line_attractor(
                cond_avg, 
                dict(dt=self.dt, sigma=self.sigma, ntrials=self.n_trials, eval1=eval1[i]), 
                time=self.simul_step
            )
            data[1].append(self.flatten_cond_trail(x2))

            x3, _ = self.point_attractor(
                cond_avg, 
                dict(dt=self.dt, sigma=self.sigma, ntrials=self.n_trials, a1=a1[i], a2=a2[i]), 
                time=self.simul_step
            )
            data[2].append(self.flatten_cond_trail(x3))
            
        data = [x for mtype in data for x in mtype]
        return data

    def visualization(self):
        """
        Visualize the behavior of all three attractor types.
        
        Generates a figure showing trajectories for each attractor type.
        """
        common_args = dict(dt=self.dt, sigma=self.sigma, ntrials=self.n_trials)
        x1, cond_avg = self.bistable_switch([-.1, .1], common_args, time=self.simul_step)
        x2, _ = self.line_attractor(cond_avg, common_args, time=self.simul_step)
        cond_avg2 = x2.mean(axis=1)
        x3, _ = self.point_attractor(cond_avg, common_args, time=self.simul_step)
        cond_avg3 = x3.mean(axis=1)
        
        fig, ax = plt.subplots(1, 3, figsize=(8, 5), sharex=True, sharey=True)
        for i in range(x1.shape[0]):
            for k, cavg in enumerate([cond_avg, cond_avg2, cond_avg3]):
                ax[k].plot(cavg[i,:,0], cavg[i,:,1])
            for j in range(x1.shape[1]):
                if j == 10:
                    break
                for k, x in enumerate([x1, x2, x3]):
                    ax[k].plot(x[i,j,:,0], x[i,j,:,1], c="red" if i else "blue", alpha=0.2)

        # Set common styling for all subplots
        for i, title in enumerate(["Saddle Point", "Line Attractor", "Point Attractor"]):
            ax[i].set_ylim(-0.2, 0.2)
            ax[i].set_xlabel(r"$h_1$", fontsize=12)
            ax[i].set_title(title, fontweight='bold', fontsize=14)
            ax[i].grid(True, linestyle='--', alpha=0.7)
            ax[i].spines['top'].set_visible(False)
            ax[i].spines['right'].set_visible(False)
            
        ax[0].set_ylabel(r"$h_2$", fontsize=12)
        
        # Add a legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=2, label='Condition 1'),
            Line2D([0], [0], color='red', lw=2, label='Condition 2'),
            Line2D([0], [0], color='black', lw=1, label='Mean trajectory')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), 
                  ncol=3, frameon=False, fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def flatten_cond_trail(self, data):
        """
        Reshape data by flattening condition and trial dimensions.
        
        Parameters:
            data (np.ndarray): Input data with shape (conditions, trials, steps, dimensions).
            
        Returns:
            np.ndarray: Reshaped data with shape (conditions*trials, steps, dimensions).
        """
        return data.reshape(data.shape[0] * data.shape[1], data.shape[2], data.shape[3])