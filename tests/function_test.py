from models import data_preparation, mulkde_coef, mulkde, other_kde
import pytest
import numpy as np

# test data_preparation.DistributionSampler in data_preparation
with pytest.raises(ValueError, match='Unknown distribution'):
    sampler = data_preparation.DistributionSampler('hello',[1,2])

with pytest.raises(ValueError, match='parameters'):
    data_preparation.DistributionSampler('bimodal',[1,2])

# test invert_symmetric_matrix
with pytest.raises(ValueError, match='square'):
    mulkde_coef.invert_symmetric_matrix(np.ones((1,2)))

# test kde in mulkde
with pytest.raises(TypeError, match='number'):
    mulkde.kde('a', np.ones(3))

# test adtive kde in other_kde
with pytest.raises(ValueError, match ='bandwidth'):
    other_kde.adaptive_kde(-1,np.array([1,2,3]))

print("All tests passed!")