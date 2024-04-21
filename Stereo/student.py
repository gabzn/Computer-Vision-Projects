import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo (or mean albedo for RGB input images) for a pixel is less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights. All N images have the same dimension.
                  The images can either be all RGB (height x width x 3), or all grayscale (height x width x 1).

    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    def compute_albedo(G):
        r = G.shape[0]
        # new_shape = [s for s in shape[1: ]]
        # new_shape = [r] + new_shape
        new_shape = [r]
        for s in shape:
            new_shape.append(s)

        G = G.reshape(new_shape)
        return np.linalg.norm(G, axis=0)
    
    def compute_normals(G):
        new_shape = [s for s in shape]
        new_shape.append(3)

        G_transpose = G.T
        normals = np.mean(G_transpose.reshape(new_shape), axis=2)
        norm = np.linalg.norm(normals, axis=2)

        maxes = np.maximum(norm[:, :, np.newaxis], EPSILON)
        normals = normals / maxes
        return np.nan_to_num(normals)

    N = len(images)
    EPSILON = 1e-7
    shape = np.shape(images)
    shape = shape[1: ]
    
    """
    I = L * G
    L_transpose * I = L_transpose * L * G
    G = (L_transpose * L)^-1 * (L_transpose * I)
    """
    I = np.array(images)
    product = np.prod(shape)
    I = I.reshape(N, product)

    L = lights
    L_transpose = L.T
    L_transpose_L = np.dot(L_transpose, L)
    L_transpose_L_inv = np.linalg.inv(L_transpose_L)
    L_transpose_I = np.dot(L_transpose, I)

    G = np.dot(L_transpose_L_inv, L_transpose_I)
    return compute_albedo(G), compute_normals(G)


def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    h, w, _ = points.shape
    
    projection_matrix = np.dot(K, Rt)
    projection_matrix_transpose = projection_matrix.T

    new_col = np.ones((h, w, 1))
    homo = np.concatenate((points, new_col), axis=2)
    homo = np.dot(homo, projection_matrix_transpose)

    normalized_projs = homo / homo[:, :, 2: 3]
    return normalized_projs[:, :, :2]


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc_impl' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc_impl' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    EPSILON = 1e-6

    h, w, c = image.shape
    normalized = np.zeros((h, w, (ncc_size ** 2) * c), dtype=np.float32)
    
    mid = ncc_size // 2
    for i in range(mid, h - mid):
        for j in range(mid, w - mid):
            patches = []

            for k in range(c):
                r_start = i - mid
                r_end = i + mid
                
                c_start = j - mid
                c_end = j + mid

                patch = image[r_start: r_end + 1, c_start: c_end + 1, k]
                patch_mean = (patch - np.mean(patch)).flatten()
                patch_mean_transpose = patch_mean.T
                
                patches.append(patch_mean_transpose)

            patches_flattened = np.array(patches).flatten()
            patches_flattened_norm = np.linalg.norm(patches_flattened)

            if patches_flattened_norm <= EPSILON:
                normalized[i,j] = np.zeros(patches_flattened.shape)
            else:
                normalized[i,j] = patches_flattened / patches_flattened_norm

    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc_impl.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    # cross_correlation = image1 * image2
    cross_correlation = np.multiply(image1, image2)
    return np.sum(cross_correlation, axis=2)