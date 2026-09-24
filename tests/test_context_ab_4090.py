import unittest

import torch

from scripts.context_ab_4090 import (flow_training_pair, paired_step_seed,
                                     transition_rmse, assert_matched_baseline,
                                     history_block_count)


class ContextABTests(unittest.TestCase):
    def test_flow_target_matches_linear_noising(self):
        clean = torch.ones(1, 16, 3, 2, 2)
        noise = torch.zeros_like(clean)
        xt, velocity = flow_training_pair(clean, noise, 0.25)
        self.assertTrue(torch.equal(xt, torch.full_like(clean, 0.75)))
        self.assertTrue(torch.equal(velocity, -clean))

    def test_step_seed_is_mode_independent(self):
        self.assertEqual(paired_step_seed(17, 3), paired_step_seed(17, 3))
        self.assertNotEqual(paired_step_seed(17, 3), paired_step_seed(17, 4))

    def test_transition_metric_detects_boundary_spike(self):
        video = torch.tensor([0., 0., 0., 10., 10., 10.]).view(1, 1, 6, 1, 1)
        metric = transition_rmse(video, frames_per_block=3)
        self.assertEqual(metric['boundary_rmse'], 10.0)
        self.assertEqual(metric['within_rmse'], 0.0)

    def test_baseline_assertion_rejects_unpaired_validation(self):
        same = [dict(teacher_flow_mse=0.1, self_flow_mse=0.2)]
        assert_matched_baseline(same, [dict(same[0])])
        with self.assertRaises(ValueError):
            assert_matched_baseline(same, [dict(teacher_flow_mse=0.1,
                                                self_flow_mse=0.3)])

    def test_long_clip_history_blocks(self):
        self.assertEqual(history_block_count(21), 6)
        self.assertEqual(history_block_count(48), 15)
        with self.assertRaises(ValueError):
            history_block_count(49)


if __name__ == '__main__':
    unittest.main()
