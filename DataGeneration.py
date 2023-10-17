import numpy as np

def make_random_cube(N, non_polar_frac, size=20):

    """ Returns a numpy array with random positions, APB and PCP
        Along with a mask dictating which particles are non polarized  """
    
    x = np.random.uniform(-size/2,size/2, size=(N,3))                                  #Particles positions

    p = np.random.uniform(-1,1,size=(N,3))                                             #AB polarity unit vectors
    p /= np.sqrt(np.sum(p**2, axis=1))[:,None]

    q = np.random.uniform(-1,1,size=(N,3))                                             #PCP unit vectors
    q /= np.sqrt(np.sum(p**2, axis=1))[:,None]

    mask = np.random.choice([0,1], p=[non_polar_frac, 1-non_polar_frac], size=N)       #Mask detailing which particles are non polar
    p[mask == 0] = np.array([0,0,0])                                                   #Setting the polarities of the non-polarized particles to 0
    q[mask == 0] = np.array([0,0,0])

    cube_data = np.concatenate((mask[:,None], x, p, q) ,axis=1)                        #Total data
    return cube_data