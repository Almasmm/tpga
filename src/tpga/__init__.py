"""tpga — the transportation problem solved by a genetic algorithm, with an exact LP baseline."""

from tpga.decoder import decode_permutation
from tpga.exact import solve_lp
from tpga.ga import GAConfig, GAResult, run_ga
from tpga.model import TransportProblem

__all__ = ["GAConfig", "GAResult", "TransportProblem", "decode_permutation", "run_ga", "solve_lp"]
__version__ = "0.1.0"
