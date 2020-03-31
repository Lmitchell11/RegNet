import numpy as np

def get_calib(date):
    if date == '2011_09_26':
        return CAM02_PARAMS_2011_09_26, VELO_PARAMS_2011_09_26
    elif date == '2011_09_30':
        return CAM02_PARAMS_2011_09_30, VELO_PARAMS_2011_09_30

CAM02_PARAMS_2011_09_26 = dict(
    fx = 7.215377e+02,
    fy = 7.215377e+02,
    cx = 6.095593e+02,
    cy = 1.728540e+02,
    rot = [[9.999758e-01, -5.267463e-03, -4.552439e-03],
           [5.251945e-03, 9.999804e-01, -3.413835e-03],
           [4.570332e-03, 3.389843e-03, 9.999838e-01]],
    trans = [[5.956621e-02], [2.900141e-04], [2.577209e-03]],
)

VELO_PARAMS_2011_09_26 = dict(
    rot = [[7.533745e-03, -9.999714e-01, -6.166020e-04],
           [1.480249e-02, 7.280733e-04, -9.998902e-01],
           [9.998621e-01, 7.523790e-03, 1.480755e-02]],
    trans = [[-4.069766e-03], [-7.631618e-02], [-2.717806e-01]],
)

CAM02_PARAMS_2011_09_30 = dict(
    fx = 7.070912e+02,
    fy = 7.070912e+02,
    cx = 6.018873e+02,
    cy =1.831104e+02,
    rot = [[9.999805e-01, -4.971067e-03, -3.793081e-03],
           [4.954076e-03, 9.999777e-01, -4.475856e-03],
           [3.815246e-03, 4.456977e-03, 9.999828e-01]],
    trans = [[6.030222e-02], [-1.293125e-03], [5.900421e-03]],
)

VELO_PARAMS_2011_09_30 = dict(
    rot = [[7.027555e-03, -9.999753e-01, 2.599616e-05],
           [-2.254837e-03, -4.184312e-05, -9.999975e-01],
           [9.999728e-01, 7.027479e-03, -2.255075e-03]],
    trans = [[-7.137748e-03], [-7.482656e-02], [-3.336324e-01]],
)