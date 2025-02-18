# ---   IMPORTS   --- #
# ------------------- #
from src.tests.vl3d_test import VL3DTest
from src.model.deeplearn.loss.class_weighted_binary_crossentropy import \
    vl3d_class_weighted_binary_crossentropy as _cwbc
from src.model.deeplearn.loss.class_weighted_categorical_crossentropy import \
    vl3d_class_weighted_categorical_crossentropy as _cwcc
from src.model.deeplearn.loss.ragged_binary_crossentropy import \
    vl3d_ragged_binary_crossentropy as _rbc
from src.model.deeplearn.loss.ragged_categorical_crossentropy import \
    vl3d_ragged_categorical_crossentropy as _rcc
from src.model.deeplearn.loss.ragged_class_weighted_binary_crossentropy import \
    vl3d_ragged_class_weighted_binary_crossentropy as _rcwbc
from src.model.deeplearn.loss.ragged_class_weighted_categorical_crossentropy \
    import vl3d_ragged_class_weighted_categorical_crossentropy as _rcwcc
import tensorflow as tf
import numpy as np

# ---   CLASS   --- #
# ----------------- #
class RaggedLossesTest(VL3DTest):
    """
    :author: Alberto M. Esmoris Pena

    Ragged losses test that checks the operations of loss functions work with
    ragged tensors and yield the expected results.
    """
    # ---   INIT   --- #
    # ---------------- #
    def __init__(self):
        super().__init__('Ragged losses test')
        self.eps = 1e-5

    # ---   TEST INTERFACE   --- #
    # -------------------------- #
    def run(self):
        """
        Run ragged losses test.

        :return: True if ragged losses work as expected for the test cases,
            False otherwise.
        :rtype: bool
        """
        valid = RaggedLossesTest.test_classification_losses(
            self.eps
        )
        valid = valid and RaggedLossesTest.test_ragged_classification_losses(
            self.eps
        )
        return valid

    # ---   TEST SUITES   --- #
    # ----------------------- #
    @staticmethod
    def test_classification_losses(eps):
        # Generate binary test batches
        btrue1 = np.random.randint(0, 2, 128).astype(np.float32)
        bpred1 = np.random.uniform(0, 1, btrue1.shape[0]).astype(np.float32)
        btrue2 = np.random.randint(0, 2, (11, 77)).astype(np.float32)
        bpred2 = np.random.uniform(0, 1, btrue2.shape).astype(np.float32)
        # Generate categorical test batches
        ny = 5  # Number of fake classes
        ctrue1 = np.zeros((128, ny)).astype(np.float32)
        _ = np.random.randint(0, ny, ctrue1.shape[0])
        for i in range(len(ctrue1)):
            ctrue1[i, _[i]] = 1
        cpred1 = np.random.uniform(0, 1, ctrue1.shape).astype(np.float32)
        cpred1 = (cpred1.T / np.linalg.norm(cpred1, axis=1)).T
        ctrue2 = np.zeros((11, 77, ny)).astype(np.float32)
        _ = np.random.randint(0, ny, ctrue2.shape[:2])
        _ = [
            np.random.randint(0, ny, ctrue2.shape[1])
            for k in range(ctrue2.shape[0])
        ]
        for i in range(len(ctrue2)):
            _i = _[i]
            for j in range(len(ctrue2[i])):
                ctrue2[i, j, _i[j]] = 1
        cpred2 = np.random.uniform(0, 1, ctrue2.shape).astype(np.float32)
        cpred2 = (cpred2.T / np.linalg.norm(cpred2, axis=2).T).T
        # Prepare class weighted losses
        bcw = np.random.uniform(0, 1, 2).astype(np.float32)  # Binar. cls. weig.
        ccw = np.random.uniform(0, 1, ny).astype(np.float32)  # Categor. cls. w.
        rbc = _rbc()
        cwbc = _cwbc(bcw)
        rcwbc = _rcwbc(bcw)
        rcc = _rcc()
        cwcc = _cwcc(ccw)
        rcwcc = _rcwcc(ccw)
        # Test classification losses
        with tf.device("cpu:0"):
            # Test for binary crossentropy (1)
            bcl_ref = tf.keras.losses.BinaryCrossentropy()(btrue1, bpred1)
            bcl = rbc(btrue1, bpred1)
            if np.any(np.abs(bcl_ref - bcl) > eps):
                return False
            # Test for binary crossentropy (2)
            bcl_ref = tf.keras.losses.BinaryCrossentropy()(btrue2, bpred2)
            bcl = rbc(btrue2, bpred2)
            if np.any(np.abs(bcl_ref - bcl) > eps):
                return False
            # Test for class weighted binary crossentropy (1)
            cwbcl_ref = cwbc(btrue1, bpred1)
            cwbcl = rcwbc(btrue1, bpred1)
            if np.any(np.abs(cwbcl_ref - cwbcl) > eps):
                return False
            # Test for class weighted binary crossentropy (2)
            cwbcl_ref = cwbc(btrue2, bpred2)
            cwbcl = rcwbc(btrue2, bpred2)
            if np.any(np.abs(cwbcl_ref - cwbcl) > eps):
                return False
            # Test for categorical crossentropy (1)
            ccl_ref = tf.keras.losses.CategoricalCrossentropy()(ctrue1, cpred1)
            ccl = rcc(ctrue1, cpred1)
            if np.any(np.abs(ccl_ref - ccl) > eps):
                return False
            # Test for categorical crossentropy (2)
            ccl_ref = tf.keras.losses.CategoricalCrossentropy()(ctrue2, cpred2)
            ccl = rcc(ctrue2, cpred2)
            if np.any(np.abs(ccl_ref - ccl) > eps):
                return False
            # Test for class weighted categorical crossentropy (1)
            cwccl_ref = cwcc(ctrue1, cpred1)
            cwccl = rcwcc(ctrue1, cpred1)
            if np.any(np.abs(cwccl_ref - cwccl) > eps):
                return False
            # Test for class weighted categorical crossentropy (2)
            cwccl_ref = cwcc(ctrue2, cpred2)
            cwccl = rcwcc(ctrue2, cpred2)
            if np.any(np.abs(cwccl_ref - cwccl) > eps):
                return False
        # On all checks passed
        return True

    @staticmethod
    def test_ragged_classification_losses(eps):
        # Generate ragged binary test batches
        rbtrue1 = [
            np.random.randint(0, 2, 128).astype(np.float32),
            np.random.randint(0, 2, 192).astype(np.float32),
            np.random.randint(0, 2, 56).astype(np.float32),
        ]
        rbpred1 = [
            np.random.uniform(0, 1, rbtrue1[0].shape[0]).astype(np.float32),
            np.random.uniform(0, 1, rbtrue1[1].shape[0]).astype(np.float32),
            np.random.uniform(0, 1, rbtrue1[2].shape[0]).astype(np.float32),
        ]
        rbtrue1 = tf.ragged.constant(rbtrue1)
        rbpred1 = tf.ragged.constant(rbpred1)
        rbtrue2 = [
            np.random.randint(0, 2, (11, 77)).astype(np.float32),
            np.random.randint(0, 2, (11, 55)).astype(np.float32),
            np.random.randint(0, 2, (11, 33)).astype(np.float32),
        ]
        rbpred2 = [
            np.random.uniform(0, 1, rbtrue2[0].shape).astype(np.float32),
            np.random.uniform(0, 1, rbtrue2[1].shape).astype(np.float32),
            np.random.uniform(0, 1, rbtrue2[2].shape).astype(np.float32),
        ]
        rbtrue2 = tf.ragged.constant(rbtrue2)
        rbpred2 = tf.ragged.constant(rbpred2)
        # Generate ragged categorical test batches
        ny = 5  # Number of fake classes
        rctrue1 = [
            np.zeros((128, ny)).astype(np.float32),
            np.zeros((192, ny)).astype(np.float32),
            np.zeros((56, ny)).astype(np.float32)
        ]
        rcpred1 = [
            np.random.uniform(0, 1, rctrue1[0].shape).astype(np.float32),
            np.random.uniform(0, 1, rctrue1[1].shape).astype(np.float32),
            np.random.uniform(0, 1, rctrue1[2].shape).astype(np.float32),
        ]
        for i in range(len(rctrue1)):
            for j in range(len(rctrue1[i])):
                rctrue1ij = rctrue1[i][j]
                rctrue1ij[np.random.randint(0, ny)] = 1
            rcpred1i = rcpred1[i]
            rcpred1[i] = (rcpred1i.T / np.linalg.norm(rcpred1i, axis=1)).T
        rctrue1 = tf.ragged.constant(rctrue1, ragged_rank=1)
        rcpred1 = tf.ragged.constant(rcpred1, ragged_rank=1)
        rctrue2 = [
            np.zeros((11, 77, ny)).astype(np.float32),
            np.zeros((11, 55, ny)).astype(np.float32),
            np.zeros((11, 33, ny)).astype(np.float32)
        ]
        rcpred2 = [
            np.random.uniform(0, 1, rctrue2[0].shape).astype(np.float32),
            np.random.uniform(0, 1, rctrue2[1].shape).astype(np.float32),
            np.random.uniform(0, 1, rctrue2[2].shape).astype(np.float32),
        ]
        for i in range(len(rctrue2)):
            for j in range(len(rctrue2[i])):
                for k in range(len(rctrue2[i][j])):
                    rctrue2ijk = rctrue2[i][j][k]
                    rctrue2ijk[np.random.randint(0, ny)] = 1
            rcpred2i = rcpred2[i]
            rcpred2[i] = (rcpred2i.T / np.linalg.norm(rcpred2i, axis=2).T).T
        rctrue2 = tf.ragged.constant(rctrue2, ragged_rank=2)
        rcpred2 = tf.ragged.constant(rcpred2, ragged_rank=2)
        # Prepare class weighted losses
        rbc = _rbc()
        bcw = np.random.uniform(0, 1, 2).astype(np.float32)  # Binar. cls. weig.
        rcc = _rcc()
        ccw = np.random.uniform(0, 1, ny).astype(np.float32)  # Categor. cls. w.
        cwbc = _cwbc(bcw)
        rcwbc = _rcwbc(bcw)
        cwcc = _cwcc(ccw)
        rcwcc = _rcwcc(ccw)
        # Test classification losses
        with tf.device("cpu:0"):
            # Test for ragged binary crossentropy (1)
            rbcl_ref = [
                tf.keras.losses.BinaryCrossentropy()(
                    rbtrue1[k], rbpred1[k]
                ).numpy()
                for k in range(tf.shape(rbtrue1)[0])
            ]
            rbcl = rbc(rbtrue1, rbpred1)
            if np.any(np.abs(
                    np.average(rbcl_ref, weights=[len(x) for x in rbtrue1]) -
                    rbcl
            ) > eps):
                return False
            # Test for ragged binary crossentropy (2)
            rbcl_ref = [
                tf.keras.losses.BinaryCrossentropy()(
                    rbtrue2[k], rbpred2[k]
                ).numpy()
                for k in range(tf.shape(rbtrue2)[0])
            ]
            rbcl = rbc(rbtrue2, rbpred2)
            if np.any(np.abs(
                np.average(
                    rbcl_ref,
                    weights=[np.prod(x.numpy().shape) for x in rbtrue2]
                ) - rbcl
            ) > eps):
                return False
            # Test for ragged class weighted binary crossentropy (1)
            rcwbcl_ref = [
                cwbc(rbtrue1[k], rbpred1[k]).numpy()
                for k in range(tf.shape(rbtrue1)[0])
            ]
            rcwbcl = rcwbc(rbtrue1, rbpred1)
            if np.abs(
                np.average(rcwbcl_ref, weights=[len(x) for x in rbtrue1])
                - rcwbcl
            ) > eps:
                return False
            # Test for ragged class weighted binary crossentropy (2)
            rcwbcl_ref = [
                cwbc(rbtrue2[k].numpy(), rbpred2[k].numpy()).numpy()
                for k in range(tf.shape(rbtrue2)[0])
            ]
            rcwbcl = rcwbc(rbtrue2, rbpred2)
            if np.abs(
                    np.average(
                        rcwbcl_ref,
                        weights=[np.prod(x.numpy().shape) for x in rbtrue2]
                    ) - rcwbcl
            ) > eps:
                return False
            # Test for ragged categorical crossentropy (1)
            rccl_ref = [
                tf.keras.losses.CategoricalCrossentropy()(
                    rctrue1[k], rcpred1[k]
                ).numpy()
                for k in range(tf.shape(rctrue1)[0])
            ]
            rccl = rcc(rctrue1, rcpred1)
            if np.abs(
                np.average(rccl_ref, weights=[len(x) for x in rctrue1])
                - rccl
            ) > eps:
                return False
            # Test for ragged categorical crossentropy (2)
            rccl_ref = [
                tf.keras.losses.CategoricalCrossentropy()(
                    rctrue2[k], rcpred2[k]
                ).numpy()
                for k in range(tf.shape(rctrue2)[0])
            ]
            rccl = rcc(rctrue2, rcpred2)
            if np.abs(
                np.average(
                    rccl_ref,
                    weights=[np.prod(x.numpy().shape[:-1]) for x in rctrue2]
                ) - rccl
            ) > eps:
                return False
            # Test for ragged class weighted categorical crossentropy (1)
            rcwccl_ref = [
                cwcc(rctrue1[k], rcpred1[k]).numpy()
                for k in range(tf.shape(rctrue1)[0])
            ]
            rcwccl = rcwcc(rctrue1, rcpred1)
            if np.abs(
                np.average(rcwccl_ref, weights=[len(x) for x in rctrue1])
                - rcwccl
            ) > eps:
                return False
            # Test for ragged class weighted categorical crossentropy (2)
            rcwccl_ref = [
                cwcc(rctrue2[k].to_tensor(), rcpred2[k].to_tensor()).numpy()
                for k in range(tf.shape(rctrue2)[0])
            ]
            rcwccl = rcwcc(rctrue2, rcpred2)
            if np.abs(
                np.average(
                    rcwccl_ref,
                    weights=[np.prod(x.numpy().shape[:-1]) for x in rctrue2]
                ) - rcwccl
            ) > eps:
                return False
        # On all checks passed
        return True
