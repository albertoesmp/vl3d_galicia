from src.tests.vl3d_test import VL3DTest, VL3DTestException
from src.vl3dpp import vl3dpp_loader
import os


class VL3DPPBackendTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    C++ backend test.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('VL3D++ backend test')

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run C++ backend test.

        :return: True if C++ backend is working, False otherwise.
        :rtype: bool
        """
        # Load and import
        vl3dpp_loader.vl3dpp_load(logging=False, warning=True)
        import pyvl3dpp as vl3dpp
        # Prepare working directory so C++ test_data is available
        rootdir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        os.chdir(os.path.join(rootdir, 'cpp'))
        # Run tests
        failed_count = vl3dpp.main_test()
        os.chdir(rootdir)
        if failed_count > 0:
            return False
        return True  # Test : success
