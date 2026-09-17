"""Small CPU tests, runnable in the existing online PyTorch environment."""
import unittest
import numpy as np
import torch
import pandas as pd
from Methods.Feedback2.probability_deletion import probability_curves, auc, validate_baseline
from types import SimpleNamespace
from Methods.Feedback2.probability_deletion import patch_orders, compatible_fingerprint, validate_curve_steps


class Toy(torch.nn.Module):
    def __init__(self, shift=0.):
        super().__init__()
        self.shift = shift

    def forward(self, x):
        score = (x[:, 0] * torch.linspace(0, 1, x.shape[-1], device=x.device)).mean((1, 2))
        return torch.stack((score, -score, score*.2), 1) + self.shift


class ProbabilityDeletionTests(unittest.TestCase):
    def test_tied_steps_are_audited_but_random_and_tie_free_stay_strict(self):
        scores = np.r_[np.ones(52), np.zeros(144)]
        archived = pd.DataFrame(dict(strategy=['top', 'top', 'random'],
                                     fraction_removed=[0., .3, .3], target_logit=[1., 1., 1.]))
        replay = archived.copy()
        replay.loc[1, 'target_logit'] = 3.
        checked = validate_curve_steps(replay, archived, scores, ('test',))
        tied = checked[checked.tie_boundary]
        self.assertEqual(len(tied), 1)
        self.assertFalse(tied.archived_replay_verified.iloc[0])
        for index in (0, 2):
            bad = replay.copy(); bad.loc[index, 'target_logit'] = 3.
            with self.assertRaisesRegex(RuntimeError, 'Tie-free/random deletion mismatch'):
                validate_curve_steps(bad, archived, scores, ('test',))

    def test_orders_match_original_float32_with_ties(self):
        scores = np.r_[np.zeros(59), np.linspace(0.1, 1, 137)].astype(np.float32)
        archived = scores.astype(np.float64)
        orders = patch_orders(archived, 123)
        np.testing.assert_array_equal(orders['bottom'][0], np.argsort(scores, kind='stable'))
        np.testing.assert_array_equal(orders['top'][0], np.argsort(-scores, kind='stable'))
        # CSV can contain nearby float64 values that represent the same float32.
        archived[60] = archived[59] + 1e-12
        orders = patch_orders(archived, 123)
        np.testing.assert_array_equal(orders['bottom'][0], np.argsort(archived.astype(np.float32), kind='stable'))

    def test_resume_only_recognized_validated_version(self):
        new = dict(version=5, implementation_sha256='new', checkpoint_sha256='abc')
        self.assertTrue(compatible_fingerprint(new, new))
        self.assertFalse(compatible_fingerprint(dict(new, version=4), new))
        self.assertFalse(compatible_fingerprint(dict(new, checkpoint_sha256='changed'), new))

    def test_baseline_diagnostics_preserve_tolerance(self):
        row = SimpleNamespace(target_class=0, prediction=0, unperturbed_target_logit=9.772541)
        validate_baseline(torch.tensor([9.772541, 0.]), row, ('test',))
        with self.assertRaisesRegex(RuntimeError, 'absolute difference='):
            validate_baseline(torch.tensor([9.777793, 0.]), row, ('test',))
        with self.assertRaisesRegex(RuntimeError, 'prediction replay=1'):
            validate_baseline(torch.tensor([9.772541, 10.]), row, ('test',))

    def test_shift_and_repeat_invariance(self):
        image = torch.arange(3*28*28).reshape(1, 3, 28, 28).float()/1000
        scores = np.arange(196, dtype=float)
        a, _ = probability_curves(Toy(), image, scores, 0, 123)
        b, _ = probability_curves(Toy(10), image, scores, 0, 123)
        c, _ = probability_curves(Toy(), image, scores, 0, 123, batch_size=1)
        self.assertEqual(len(a), 49)
        self.assertTrue(np.allclose(a.probability_drop,b.probability_drop,atol=1e-6))
        self.assertTrue(np.allclose(a.probability_drop,c.probability_drop,atol=1e-6))
        self.assertTrue(np.allclose(a.loc[a.fraction_removed.eq(0),"probability_drop"],0))
        self.assertEqual(set(a[a.strategy.eq("random")].repetition),set(range(5)))
        self.assertEqual(auc(a[a.strategy.eq("top")],"probability_drop"),
                         auc(a[a.strategy.eq("top")].iloc[::-1],"probability_drop"))

    def test_bad_map_rejected(self):
        with self.assertRaises(ValueError):
            probability_curves(Toy(),torch.zeros(1,3,28,28),np.zeros(195),0,1)


if __name__ == "__main__":
    unittest.main()
