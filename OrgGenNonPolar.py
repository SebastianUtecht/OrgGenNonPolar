### Imports ###
import numpy as np
import torch
from scipy.spatial import cKDTree
import scipy.io
import os
import itertools
import gc

### Setting device ###
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == 'cuda':
    print('Using cuda')
    float_tensor = torch.cuda.FloatTensor
else:
    float_tensor = torch.FloatTensor

### Defining the simulation ###

## Function for preparing data ##
def prepare_data(str):

    """ Function that prepares the data for us. 
    I.e. gets it and casts it to the appropriate tensor types

    Input: str of a .npy file containing bool mask of polar and non polar particles,
    positions, ABP and PCP vectors.
    Output: torch.tensors of the above on the initialized device.
    """
    
    #Importing data 
    data = np.load(str)                                                                          # We load the data
    p_mask = data[:,0]                                                                           # Boolean mask of non-polar and polar particles. 0:non polar, 1:polar.
    x = data[:,1:4]                                                                              # Positions
    p = data[:,4:7]                                                                              # ABP vectors
    q = data[:,7:10]                                                                             # PCP vectors

    #Casting all the data to the right torch tensors
    x = torch.tensor(x, requires_grad=True, dtype=torch.float, device=device)
    p = torch.tensor(p, requires_grad=True, dtype=torch.float, device=device)
    q = torch.tensor(q, requires_grad=True, dtype=torch.float, device=device)
    p_mask = torch.tensor(p_mask, dtype=torch.int, device=device)

    return x, p, q, p_mask

## Functions for determining the particles that interact with each other ##
def find_potential_neighbours(x, k, distance_upper_bound=np.inf):

    """Function that finds the potential nearest neighbors of all particles

    Input:
        x : numpy array (N,3) (N is number of particles)
            numpy array containing the 
        k : int
            How many possible nearest neighbors do we find pr. particle?
    
    Output:
        d   : numpy array (N,k)
              The distances from each potential nearest neighbor to the particle
        idx : numpy array (N,k)
              Indexes of the potential nearest neighbors
    """

    tree = cKDTree(x)                                                                             # Making a tree structure to find nearest neighbors quick
    d, idx = tree.query(x, k + 1, distance_upper_bound=distance_upper_bound, workers=-1)          # Quering the tree we just made for k nearest neighbors 
    return d[:, 1:], idx[:, 1:]                                                                   # Returning distances and the indexes of the found k nearest neighbors
                                                                                                  # The columns taken out are self-referential. (nearest neighbor to each point is itself.)

def find_true_neighbours(d, dx):

    """Function that takes potential neighbors and determines whether they interact
    through voronoi criteria

    Input:
        d : torch tensor (N,k)
            The distances from each potential nearest neighbor to the particle
        dx : torch tensor (N,k,3)
             Tensor containing vectors pointing from each potential neighbor to the particle
    Output:
        z_mask : torch tensor bool (N,k)
                 Boolean mask that can be applied to the potential neighbor indexes and distances
                 to filter out particles that shouldn't interact.
    """                                                                                           
    
    with torch.no_grad():                                                                         # We don't need to calculate gradients here, so they are 'thrown away'. Lessens memory consumption.
        z_masks = []                                                                              # list that will contain the bool mask
        i0 = 0                                                                                    # We do the following calculation in batches as the tensors get quite big
        batch_size = 250                   
        i1 = batch_size
        while True:
            if i0 >= dx.shape[0]:
                break

            n_dis = torch.sum((dx[i0:i1, :, None, :] / 2 - dx[i0:i1, None, :, :]) ** 2, dim=3)    # Finding distances from particle k to the midpoint between particles i and j squared for all potential neighbors
            n_dis += 1000 * torch.eye(n_dis.shape[1], device=device)[None, :, :]                  # We add 1000 to the sites that correspond to subtracting the vector ij/2 with ij, so these don't fuck anything up.

            z_mask = torch.sum(n_dis < (d[i0:i1, :, None] ** 2 / 4), dim=2) <= 0                  # If all the distances from particles k to midpoint ij/2 are bigger than half the distance from i to j we use the connection ij.
            z_masks.append(z_mask)                                                                # Boolean mask that can be applied to the idx (from the tree query) to ascertain whether they are true neighbors

            if i1 > dx.shape[0]:                                                                  # We stop when we run out of stuff to do (duh)
                break
            i0 = i1
            i1 += batch_size
    z_mask = torch.cat(z_masks, dim=0)                                                            # Converting to torch tensor
    return z_mask


## Function for calculating potential ##
def potential(dx, p, q, idx, d, p_mask, z_mask, l00, l01, l1, l2, l3):

    """Function that calculates the potential of the particle configuration

    Input:
        dx : torch tensor (N,m,3) (m is the maximal number of true neighbor any particle in our system has)
             Tensor containing vectors pointing from each potential neighbor to the particle
        p  : torch tensor (N,3)
             Tensor containing the ABP vectors
        q  : torch tensor (N,3)
             Tensor containing the PCP vectors
        idx : torch tensor (N,m)
              Indicies containing mostly true neighbors (some false have survived but will be killed by z_mask later)
        p_mask : torch tensor bool (N)
                 Tensor containing boolean mask of which particles are polarized and which isn't
        z_mask : torch tensor bool (N,m)
                 Boolean mask that can be applied to the potential neighbor indexes and distances
                 to filter out particles that shouldn't interact.
        l's    : floats
                 The different lambda values that dictate how much each term of S is weighted
    Output:
        V  : torch tensor (1)
             Scalar value of the potential
    """

    # Calculate S
    pi = p[:, None, :].expand(p.shape[0], idx.shape[1], 3)                                        # We expand the ABP tensor in order to be able to do all cross products in 1 go
    pj = p[idx]                                                                                   # For each particle we get the ABP of its nearest neighbors
    
    qi = q[:, None, :].expand(q.shape[0], idx.shape[1], 3)                                        # Expansion for later cross product
    qj = q[idx]                                                                                   # For each particle we the the PCP of its nearest neighbors
    
    
    interaction_mask = p_mask[:,None].expand(p_mask.shape[0], idx.shape[1]) + p_mask[idx]         # We make a mask of all particle interactions. 0 = completely non polar interaction. 1 = polar - nonpolar interaction. 2 = completely polar interaction.
    lam = torch.zeros(size=(interaction_mask.shape[0], interaction_mask.shape[1], 4),
                       device=device)                                                             # Initializing an empty array for our lambdas
    lam[interaction_mask == 0] = torch.tensor([l00,0,0,0], device=device)                         # Setting lambdas for non polar interaction
    lam[interaction_mask == 1] = torch.tensor([l01,0,0,0], device=device)                         # Setting lambdas for polar-nonpolar interaction
    lam[interaction_mask == 2] = torch.tensor([0,l1,l2,l3], device=device)                        # Setting lambdas for pure polar interaction
    lam.requires_grad = True                                                                      # We need these gradients in order to do backprob later.


    S1 = torch.sum(torch.cross(pj, dx, dim=2) * torch.cross(pi, dx, dim=2), dim=2)                # Calculating S1 (The ABP-position part of S). Scalar for each particle-interaction. Meaning we get array of size (n, m) , m being the max number of nearest neighbors for a particle
    S2 = torch.sum(torch.cross(pi, qi, dim=2) * torch.cross(pj, qj, dim=2), dim=2)                # Calculating S2 (The ABP-PCP part of S).
    S3 = torch.sum(torch.cross(qi, dx, dim=2) * torch.cross(qj, dx, dim=2), dim=2)                # Calculating S3 (The PCP-position part of S)

    S = lam[:,:,0] + lam[:,:,1] * S1 + lam[:,:,2] * S2 + lam[:,:,3] * S3                          # Calculating S total. Weighing each interaction by their appropriate lambas. array shape (n,m)

    # Potential
    Vij = z_mask.float() * (torch.exp(-d) - S * torch.exp(-d/5))                                  # Calculating the potentials between all (nearest neighbor) particles
    V = torch.sum(Vij)                                                                            # Calculating full potential for downstream backpropagation

    return V                                                                                      # We return full potential and max number of nearest neighbors for a particle

## Class for progressing simulation  ##
class TimeStepper:

    """Class used to progress our progress the simulation"""

    def __init__(self, init_k):                                                                   # We save some stuff for later use    
        """
        Inputs:
            init_k : int
                     How many initial potential neighbors do we want?
        """
        self.k = init_k
        self.true_neighbour_max = init_k//2
        self.d = None
        self.idx = None

    def update_k(self, true_neighbour_max, tstep):

        """Function for dynamically changing k (number of potential nearest neighbors)
        so we don't waste ressources
        
        Inputs:
            true_neighbor_max : int
                               Maximum number of true neighbors any one particle has. Other places
                               referred to as m.
            tstep : int
                    The timestep we are currently on.
        
        Outputs:
            k : int
                Updated k
            n_update : int
                       Variable used for determining when we find new potential nearest neighbors,
                       As finding potential nearest neighbors is an expensive operation due to memory
                       allocation.
        """
        
        k = self.k
        fraction = true_neighbour_max / k                                                         # Fraction between the maximimal number of nearest neighbors and the initial nunber of nearest neighbors we look for.
        if fraction < 0.25:                                                                       # If fraction is small our k is too large and we make k smaller
            k = int(0.75 * k)
        elif fraction > 0.75:                                                                     # Vice versa
            k = int(1.5 * k)
        n_update = 1 if tstep < 50 else max([1, int(20 * np.tanh(tstep / 200))])                  # We don't find new neighbors for the first 50 timesteps. Afterwards we do when the tanh function 'allows'. Meaning we update frequently just after tstep=50 and then less and less frequent until we only update every 20. timestep
        self.k = k                                                                                # We update k
        return k, n_update

    def time_step(self, x, p, q, p_mask, tstep, dt, sqrt_dt, eta, l00, l01, l1, l2, l3):

        """Function that updates our system

        Inputs:
            x : torch tensor (N,3)
                Tensor containing positions of all particles
            p : torch tensor (N,3)
                Tensor containing the ABP vectors
            q : torch tensor (N,3)
                Tensor containing the PCP vectors
            p_mask : torch tensor bool (N)
                     Tensor containing boolean mask of which particles are polarized and which isn't
            tstep : int
                    Which timestep are we on?
            dt : float
                 The time increment we update our system with
            sqrt_dt : float
                      Squareroot of dt. Used for updating our noise.
            eta : float
                  Parameter for updating our noise (ASK!)
            l's : floats
                  The different lambda values that dictate how much each term of S is weighted
        """

        # Idea: only update _potential_ neighbours every x steps late in simulation
        # For now we do this on CPU, so transfer will be expensive

        # Making assertions to make sure we ain't fucking something up
        assert l1 + l2 + l3 == 1                                                                  # Checking 'normalization' requirement of the lambas
        assert q.shape == x.shape                                                                 # Checking whether the shapes of our positions and polarizations are as they should be
        assert x.shape == p.shape

        # Finding potential neighbors at specific timesteps
        k, n_update = self.update_k(self.true_neighbour_max, tstep)                               # We update k and find n_update
        if tstep % n_update == 0 or self.idx is None:                                             # If n_update is the right value we find potential neighbors again 
            d, idx = find_potential_neighbours(x.detach().to("cpu").numpy(), k=k)                 # Find potential neighbors and their distances to the the queried particles
            self.idx = torch.tensor(idx, dtype=torch.long, device=device)                         # Update indices of potential neighbors and cast to torch tensor
            self.d = torch.tensor(d, dtype=torch.float, device=device)                            # Update distances of potential neighbors and cast to torch tensor
        idx = self.idx
        d = self.d                                                                                # IS THIS NECESSARY? I THINK WE OVERWRITE LATER / DONT USE

        # Normalise p, q
        with torch.no_grad():
            p[p_mask != 0] /= torch.sqrt(torch.sum(p[p_mask != 0] ** 2, dim=1))[:, None]          # Normalizing p. Only the non-zero polarities are considered.
            q[p_mask != 0] /= torch.sqrt(torch.sum(q[p_mask != 0] ** 2, dim=1))[:, None]          # Normalizing q. Only the non-zero polarities are considered.
            p[p_mask == 0] = torch.tensor([0.,0.,0.], device=device)                              # Make sure that our non-polarized particles stay non-polarized. I don't think this is strictly necessary, but it's nice to be sure.
            q[p_mask == 0] = torch.tensor([0.,0.,0.], device=device)                              # Same as above

        
        # Find true neighbours
        full_n_list = x[idx]                                                                      # Tensor containing the coordinates for all the potential neighbors of each particle (N,k,3)
        dx = x[:, None, :] - full_n_list                                                          # Tensor containing vectors pointing from each particles potential neighbor to it    (N,k,3)
        z_mask = find_true_neighbours(d, dx)                                                      # Finding boolean mask that can be used on idx to find true (N,k)
        z_mask = z_mask.int()                                                                     # We cast the boolean mask to integers. Otherwise pytorch doesn't like us :(
        
        # Minimize size of z_mask and reorder idx and dx
        sort_idx = torch.argsort(z_mask, dim=1, descending=True)                                  # We find sort-indices that put all the 1's in our mask before all the 0's. We this so we can throw away as many non-used potential neighbors as possible.
        z_mask = torch.gather(z_mask, 1, sort_idx)                                                # We use the sorted indices we just found on the mask
        dx = torch.gather(dx, 1, sort_idx[:, :, None].expand(-1, -1, 3))                          # We use the sorted indices on the dx tensor (some broadcasting necessary)
        idx = torch.gather(idx, 1, sort_idx)                                                      # We use the sorted indices we just found on the potential neighbor indices
        m = torch.max(torch.sum(z_mask, dim=1)) + 1                                               # We sum along each of the possible neighbor bool masks and note the biggest one. This value is the maximal number of nearest neighbors that any particle has
        z_mask = z_mask[:, :m]                                                                    # We throw away as many zero's as we can. Corresponding to throwing away potential neighbors of particles that are not used due to voronoi conditions.
        dx = dx[:, :m]                                                                            # Same as above
        idx = idx[:, :m]                                                                          # Same as above
        self.true_neighbour_max = m                                                               # We update the maximal number of true neighbors
        
        # Normalize dx
        d = torch.sqrt(torch.sum(dx**2, dim=2))                                                   # We redefine d and normalize the dx's. We redefine so these new d's correspond with the sorting we have done just above.
        dx = dx / d[:, :, None]
        
        # Calculate potential
        V = potential(dx=dx, p=p, q=q, idx=idx, d=d, p_mask=p_mask, z_mask=z_mask, l00=l00, l01=l01, l1=l1, l2=l2, l3=l3) # We get the total potential

        # Backpropagation
        V.backward()                                                                              # Backpropagation. By the chain rule, all gradients are found and stored in their respective tensors 

        # Time-step
        with torch.no_grad():                                                                     # We use nograds in order not to fuck with the computational graph (I think)
            x += -x.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt                  # Update our system by overdamped dynamics
            p += -p.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt                  # Same as above
            # q is kept fixed
            #q += -q.grad * dt + eta * float_tensor(*x.shape).normal_() * sqrt_dt                 # We don't change our PCP in this simulation (IMPLEMENT WITH BOOL INPUT)

        # Zero gradients
        x.grad.zero_()                                                                            # Zero out the gradient so we can start again
        p.grad.zero_()
        q.grad.zero_()

        # return x, p, q                                                                          # I COMMENT OUT THIS RETURN STATEMENT FOR NOW, AS IT SEEMS IT ISN'T NEEDED. WE INSTEAD UPDATE 'GLOBAL VARIABLES' x, p, AND q


## Iterable function that simulates our system ##
def simulation(x, p, q, p_mask, dt, eta, l00, l01, l1, l2, l3, yield_every, init_k=200):

    """Function that simulates multiple timesteps

    Inputs: 
        x : torch tensor (N,3)
        Tensor containing positions of all particles
        p : torch tensor (N,3)
            Tensor containing the ABP vectors
        q : torch tensor (N,3)
            Tensor containing the PCP vectors
        p_mask : torch tensor bool (N)
                 Tensor containing boolean mask of which particles are polarized and which isn't
        dt : float
             The time increment we update our system with each timestep
        eta : float
              Parameter for updating our noise (ASK!)
        l's : floats
              The different lambda values that dictate how much each term of S is weighted
        yield_every : int
                      How many timestepps must pass between each datayield?
        init_k : int
                 How many potential nearest neighbors do we look for initially?
    Output:
        xx : numpy array (N,3)
             Positions at present time step
        pp : numpy array (N,3)
             ABP vectors at present time step
        qq : numpy array (N,3)
             PCP vectors at present time step   
    """

    sqrt_dt = np.sqrt(dt)                                                                         # Squareroot of our dt. This is for updating the noise term and we don't want to calculate it each iteration.
    time_stepper = TimeStepper(init_k=init_k)                                                     # We initialize our timestepper class with the initial potential nearest neighbor number set to init_k
    tstep = 0                                                                                     # Start at timestep 0  
    while True:
        tstep +=1                                                                                 # Add one to our timestep
        time_stepper.time_step(x=x, p=p, q=q, p_mask=p_mask, tstep=tstep, dt=dt, 
                                sqrt_dt=sqrt_dt,eta=eta, l00=l00, l01=l01, l1=l1,
                                l2=l2, l3=l3)                                                     # We run one timestep of the simulation. Running this updates the global variables x, p and q, which we save below.

        if tstep % yield_every == 0:                                                              # Every yield_every timestep we save the position and polarities for our system
            xx = x.detach().to("cpu").numpy()                                                     # We detach our data from the GPU in order to save it
            pp = p.detach().to("cpu").numpy()
            qq = q.detach().to("cpu").numpy()
            yield xx, pp, qq                                                                      # Yield is basically a return statement that does not terminate the function but lets it run on

        gc.collect()                                                                              # Memory clean up.


### Function that actually runs the simulation ###
def run_simulation(data_str, output_folder,  dt, eta, yield_steps, yield_every,
                    l00, l01, l1, l2, l3, init_k=200):
    
    """ Function we call to run the simulation

    Inputs:
        data_str : str
                   Path to an .npy file that contains p_mask, x, p, and q
                   in that order in its columns
        output_folder : str
                        Path to the folder we want to put our data in. 
                        If it does not exist, it will be created
        dt : float
             The time increment we update our system with each timestep
        eta : float
              Parameter for updating our noise (ASK!)
        yield_steps : int
                      How many times do we want to simulation to yeild data?
        yield_every : int
                      How many timestepps must pass between each datayield?
                      Note: yield_steps * yield_every = total number of timesteps simulated
        l's : floats
              The different lambda values that dictate how much each term of S is weighted
        init_k : int
                 How many potential nearest neighbors do we look for initially? 
    Outputs:
        Saves yield_steps .mat files containing the positions, ABP vectors and PCP vectors at
        that timestep.
    """

    x, p, q, p_mask = prepare_data(data_str)                                                      # We load the data and cast it to the right torch tensors
    
    try:                                                                                          # Finding or creating the output folder
        os.mkdir(output_folder)
    except OSError:
        pass
    
    i = 0                                                                                         # To keep track of timesteps

    scipy.io.savemat( output_folder + f'/t{i*yield_every}.mat',                                   # We save the initial configuration of positions, ABP and PCP
                      dict(x=x.detach().to("cpu").numpy(), 
                           p=p.detach().to("cpu").numpy(),
                           q=q.detach().to("cpu").numpy()))
    
    np.save(output_folder + '/p_mask', p_mask.detach().to("cpu").numpy())                         # Saving the nonpolar-polar boolean mask
    for xx, pp, qq in itertools.islice(simulation(x=x, p=p, q=q, p_mask=p_mask, dt=dt, eta=eta,   # We do the simulation. Each time timestep % yield_every == 0 we save the data.
                                                   l00=l00,l01=l01, l1=l1, l2=l2, l3=l3,
                                                   yield_every=yield_every,init_k=init_k),
                                                   yield_steps):
        i += 1
        print(f'Running {i*yield_every} of {yield_steps*yield_every} timesteps', end='\r')        # Keeping track of where we are timestep-wise
        scipy.io.savemat( output_folder + f'/t{i*yield_every}.mat', dict(x=xx,p=pp,q=qq))         # Saving our data
    print(f'Simulation done, saved {yield_steps+1} datapoints')                                   # Print when simulation has terminated
    return None