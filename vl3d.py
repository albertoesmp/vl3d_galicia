"""
:author: Alberto M. Esmoris Pena

The main entry point for the execution of the VL3D software.
"""


# ---   MAIN   --- #
# ---------------- #
if __name__ == '__main__':
    # Disable tensorflow messages
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Load C++ extensions
    from src.vl3dpp import vl3dpp_loader
    vl3dpp_loader.vl3dpp_load(logging=False, warning=True)

    # Call main
    from src.main.main import main
    main(rootdir=os.path.dirname(__file__))
