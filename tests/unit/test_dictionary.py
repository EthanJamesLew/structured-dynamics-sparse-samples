import numpy as np
from src.sparse_cyclic import legendre_dictionary

def test_dictionary_3():
    """this tests the pyhton implementation directly against the Matlab one for p=3"""
    import scipy.io
    import pathlib
    this_dir = pathlib.Path(__file__).parent.resolve()
    ws = scipy.io.loadmat(str(this_dir / "../matlab/dictionary_test_1.mat"))

    Dmon, Dleg, Ind1, Ind20, Ind11, Ind300, Ind210, Ind120, Ind11 = legendre_dictionary(ws['U1'], ws['p'][0][0], ws['r'][0][0])

    assert np.all((ws['Ind1'] - 1 - Ind1) == 0)
    assert np.all((ws['Ind20'] - 1 - Ind20) == 0)
    assert np.all((ws['Ind11'] - 1 - Ind11) == 0)
    assert np.all((ws['Ind300'] - 1 - Ind300) == 0)
    assert np.all((ws['Ind210'] - 1 - Ind210) == 0)
    assert np.all((ws['Ind120'] - 1 - Ind120) == 0)
    assert np.all((ws['Ind11'] - 1 - Ind11) == 0)
    assert np.all(np.abs(ws['Amon'] - Dmon) < 1E-10)
    assert np.all(np.abs(ws['Aleg'] - Dleg) < 1E-10)
