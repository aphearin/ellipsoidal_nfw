import numpy as np
from halotools.utils.vector_utilities import angles_between_list_of_vectors
from ellipsoidal_nfw import random_nfw_ellipsoid
from halotools.sim_manager import HaloTableCache
from halotools.sim_manager import CachedHaloCatalog
from halotools_ia.correlation_functions import ed_3d, ee_3d
from halotools.utils import rotate_vector_collection

from halotools.utils.vector_utilities import (elementwise_dot, elementwise_norm, normalized_vectors,
                                              angles_between_list_of_vectors, vectors_normal_to_planes)

# This is not installed!!! Right now, these files are just sitting in the same directory
import inertia_tensors as itens

def downsample_halo(halo, remaining_portion):
    """
    Assuming the halo is a numpy array of (x,y,z) coordinates, return a subset of that list with a portion taken out
    
    Params:
        halo                    - numpy array of (x,y,z) coordinates representing the dark matter particle locations
        remaining_portion       - Fraction of the total halo to keep (0.8 returns 80% of the passed in halo)
    Returns:
        downsampled             - The portion of the halo remaining
    """
    num_els = int( len(halo)*remaining_portion )
    inds = np.random.choice( np.arange(len(halo)), num_els, replace=False)
    return halo[inds]

def calculate_axes(halos):
    """
    Given a downsampled halo (or several halos), calculate a major axis using the inertia tensor
    
    Params:
        halos                   - Single halo or list of several halos where each halo is a list of (x,y,z) tuples representing cooridnates
        separate_axes           - By default, inertia_tensors returns a 6-tuple where each element is an array
                                  The first three arrays hold the major, intermediate, and minor axes
                                  The last three hold the eigenvectors
                                  Setting this to True (it's default value) will separate the axes and eigenvectors into separate arrays
    Returns:
        results                 - The 6-tuple of axes and eigenvectors (each tuple element is an array representing one property)
                                  Each array in the tuple holds a value for every inertia tensor
    """
    I = itens.iterative_inertia_tensors_3D(halos)
    return itens._principal_axes_3D(I)

def separate_axes(principal_axes):
    """
    Splits the results of calculate_axes into easier to parse arrays
    
    Params:
        principal_axes          - The 6-tuple returned by inertia_tensors._principal_axes
    
    Returns:
        axis_lengths            - an nx3 numpy array where each row is a new object and the three columns are the major, intermediate, and minor axis lengths
        evecs                   - an nx3x3 array where the first index is the object, then each of the three columns is an eigenvector for that object
    """
    axis_lengths = np.array( [ principal_axes[0], principal_axes[1], principal_axes[2] ] ).T
    evecs = np.array( [ principal_axes[3].T, principal_axes[4].T, principal_axes[5].T ] ).T
    
    return axis_lengths, evecs

def get_angles(evecs_A, evecs_B, ignore_antiparallel=True):
    """
    Simply calls angles_between_list_of_vectors between just the major eigenvectors of each set given
    
    Params:
        evecs_A                 - The list of eigenvectors from one system (multiple objects each with their given eigenvectors)
        evecs_B                 - The list of eigenvectors from the second system
        ignore_antiparallel     - if pi is the same as 0 in terms of misalignment, take the min of angle and np.pi-angle for each angle
    
    Returns:
        angles                  - The angles between each major eigenvector in evecs_A and the corresponding major eigenvector in evecs_B
    """
    
    # Get just the major eigenvectors for each sample
    major_A = evecs_A[:,:,0]
    major_B = evecs_B[:,:,0]
    
    angles = angles_between_list_of_vectors(major_A, major_B)
    
    if ignore_antiparallel:
        # If antiparallel is the same as parallel, just report the smaller angle
        angles = np.minimum( angles, np.pi-angles )
    
    return angles

# Get theta and phi from the eigenvector (with respect to the z axis)
def theta_phi(coords):
    r = np.sqrt( np.sum( coords**2, axis=1 ) )
    units = coords/np.array([r,r,r]).T
    thetas = np.arccos( units[:,2] )
    phis = np.arccos( units[:,0]/np.sin(thetas) )
    return thetas, phis

# Taken from ia_model_components.SubhaloAlignment
def get_rotation_matrix(a, b):
    r"""
    Returns the rotation matrix (only 3D) needed to rotate a into b

    Parameters
    ==========
    a : 3 element numpy array representing a vector in 3D
    b : 3 element numpy array representing a vector in 3D

    Returns
    =======
    mat : 3x3 numpy array representing the rotation matrix needed to rotate a in the direction of b
    """
    unit_a = normalized_vectors(a)
    unit_b = normalized_vectors(b)
    v = np.cross(unit_a,unit_b)[0]
    s = np.linalg.norm(v)                           # Sin of the angle
    c = np.dot(unit_a,unit_b.T)                  # Cos of the angle

    vx = np.zeros((3,3))
    vx[0,1] = -v[2]
    vx[1,0] = v[2]
    vx[0,2] = v[1]
    vx[2,0] = -v[1]
    vx[1,2] = -v[0]
    vx[2,1] = v[0]

    mat = np.eye(3) + vx + np.dot(vx,vx)*((1-c)/(s*s))
    return mat

def extract_halo_properties(halo_table):
    properties = np.array( [ halo_table["halo_b_to_a"], halo_table["halo_c_to_a"], halo_table["halo_axisA_x"], halo_table["halo_axisA_y"], \
                                        halo_table["halo_axisA_z"] ] )
    properties = np.vstack( [ np.ones(len(halo_table)), properties ] ).T
    return properties

def rotate_halos(axes_A, axes_B, halos):
    """
    Treating each point in the halo like a position vector, find the rotation matrix to take each vector in axes_A to
    the corresponding vector in axes_B. if one of these lists only has a single element, use that for every element in the
    corresponding list. Rotate each point in each halo and return the result.
    
    Params:
        axes_A                             - List of shape (n, 3) axes to be rotated. if n != npts, n must be 1
        axes_B                             - List of shape (npts, 3) axes to be rotated into
        halos                              - List of shape (npts, resolution) of (x,y,z) points to be treated as position vectors
    
    Returns:
        rotated_halos                      - Halos where each point has be rotated in the same way that took its original axis_B
                                             into the corresponding axis_A
    """
    
    R = []
    if len(axes_A) == 1:
        R = [ get_rotation_matrix(axes_A[0], axes_B[i]) for i in range(len(axes_B)) ]
    else:
        R = [ get_rotation_matrix(axes_A[i], axes_B[i]) for i in range(len(axes_B)) ]
        
    return [ rotate_vector_collection( [ R[i] ], halos[i] ) for i in range(len(halos)) ]

def generate_matching_halos(halo_properties, resolution, rotate=True):
    """
    Generates halos as lists of (x,y,z) points based on the halo properties given
    
    Params:
        halo_properties                    - (npts,ndim) list where each row is a halo.
                                             The ndim values used to make the halo are: a, b, c, major_axis_x, major_axis_y, major_axis_z
        resolution                         - The number of particles to generate for the halo
    
    Returns:
        halos                              - (npts) list where each element is a list of length (resolution) of (x,y,z) points
                                             Each halo is generated using a,b,c and rotated using the rotation matrix that would
                                             rotate the z axis [0,0,1] into the major_axis given in halo_properties
    """
    halos = []
    true_major_axes = []
    if isinstance(resolution, int):
        resolution = (np.ones(len(halo_properties))*resolution).astype(int)
    for i in range(len(halo_properties)):
        
        a, b, c, Ax, Ay, Az = halo_properties[i]
        conc = np.zeros(resolution[i])+5.
        # The placing of a,b,c makes z the major axis, x the intermediate, and y the minor
        x, y, z = random_nfw_ellipsoid(conc, a=b, b=c, c=a)
        coords = np.array( [ coord for coord in zip(x,y,z) ] )
        
        halos.append( coords )
        true_major_axes.append( [Ax, Ay, Az] )
    
    if rotate:
        halos = rotate_halos([ [0,0,1] ], true_major_axes, halos)
    
    return halos

def get_halo_center(halo):
    return np.mean(halo, axis=0)

# Eliminate halos with 0 for halo_axisA_x(,y,z)
def mask_bad_halocat(halocat):
    bad_mask = (halocat.halo_table["halo_axisA_x"] == 0) & (halocat.halo_table["halo_axisA_y"] == 0) & (halocat.halo_table["halo_axisA_z"] == 0)
    bad_mask = bad_mask ^ np.ones(len(bad_mask), dtype=bool)
    halocat._halo_table = halocat.halo_table[ bad_mask ]