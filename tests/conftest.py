import pytest

from tpga import TransportProblem

# Optimum 225, verified by the potentials method: basis x13=20, x21=5, x22=25, x31=20, x12=0;
# potentials u=(2,0,-3), v=(5,4,1); every reduced cost c_ij - u_i - v_j is nonnegative.
SMALL = {"supply": [20, 30, 20], "demand": [25, 25, 20], "cost": [[8, 6, 3], [5, 4, 9], [2, 7, 6]]}
SMALL_OPT = 225.0


@pytest.fixture
def small() -> TransportProblem:
    return TransportProblem.from_dict(SMALL)
