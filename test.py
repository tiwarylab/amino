import amino
import numpy as np

dat = np.loadtxt("data/BOUND_COLVAR").T
ops = [amino.OrderParameter(f"test_OP_{i}", d) for i, d in enumerate(dat)]
memo = amino.DistanceMatrix(50, 0.02, "epanechnikov")
memo.initialize_distances(ops)
OP_1 = ops[0]
OP_2 = ops[1]

def equal(a, b):
    return abs(a - b) < 1e-10

def test_load_data():
    '''
    Test the data loading
    '''
    assert dat.shape == (429, 1001)

def test_distance_1():
    '''
    Test the distance between an order parameter and itself
    '''

    self_dist = memo.distance(OP_1, OP_1)

    assert equal(self_dist, 0)

def test_distance_2():
    '''
    Test the distance between two order parameters to known value
    '''

    dist = memo.distance(OP_1, OP_2)

    assert equal(dist, 0.5353485087893664)


def test_distance_3():
    '''
    Test the distance cache mechanism
    '''

    dist = memo.distance(OP_1, OP_2)
    dist_same_order = memo.distance(OP_1, OP_2)
    dist_reverse_order = memo.distance(OP_1, OP_2)

    assert equal(dist, dist_same_order)
    assert equal(dist, dist_reverse_order)

def test_add_op_1():
    '''
    Test adding OPs
    '''

    mat = amino.DissimilarityMatrix(2, memo)

    mat.add_OP(OP_1)
    assert equal(mat.matrix[0][0], 0)

def test_add_op_2():
    '''
    Test adding OPs, basic symmetry
    '''

    mat = amino.DissimilarityMatrix(2, memo)

    mat.add_OP(OP_1)
    mat.add_OP(OP_2)

    dis_matrix = np.array(mat.matrix)

    assert equal(dis_matrix[1,0], dis_matrix[0,1])
    assert equal(dis_matrix[1,0], 0.5353485087893664)
    assert equal(dis_matrix[0,0], 0)
    assert equal(dis_matrix[1,1], 0)

def test_add_op_3():
    '''
    Test adding OPs to 2x2 matrix, adding the same coordinate twice
    '''

    mat = amino.DissimilarityMatrix(2, memo)

    mat.add_OP(OP_1)
    mat.add_OP(OP_1)

    assert np.array_equal(mat.matrix, [[0, 0], [0, 0]])

def test_add_op_4():

    mat = amino.DissimilarityMatrix(2, memo)

    mat.add_OP(OP_1)
    mat.add_OP(OP_1)
    mat.add_OP(OP_2)    

    dis_matrix = np.array(mat.matrix)

    assert np.array_equal(dis_matrix, [[0, 0.5353485087893664], [0.5353485087893664, 0]])

def test_initialization():

    assert equal(memo._distance_kernel(0, 1), 0.5353485087893664)
    assert equal(memo.distance(ops[0], ops[1]), 0.5353485087893664)