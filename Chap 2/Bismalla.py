from typing import List, Callable
import numpy as np

# A Function takes in an ndarray as an argument and produces an ndarray
Array_Function = Callable[[np.ndarray], np.ndarray]
# A Chain is a list of functions
Chain = List[Array_Function]

def chain_length_2(chain: Chain, a: np.ndarray) -> np.ndarray:
    """
    Evaluates two functions in a row, in a "Chain".
    """
    assert len(chain) == 2, "Length of input 'chain' should be 2"
    
    f1, f2 = chain  # Unpacking for readability
    return f2(f1(a))  # Use 'a' instead of 'x'
