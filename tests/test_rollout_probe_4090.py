import unittest

import torch
from torch import nn

from scripts.rollout_probe_4090 import LoRALinear, make_kv_cache, renoise


class ProbeUnitTests(unittest.TestCase):
    def test_renoise_endpoints_and_detachment(self):
        clean = torch.ones(1, 16, 1, 4, 4, requires_grad=True)
        noise = torch.zeros_like(clean)
        self.assertTrue(torch.equal(renoise(clean, noise, 0.0), clean.detach()))
        self.assertTrue(torch.equal(renoise(clean, noise, 1.0), noise))
        middle = renoise(clean, noise, 0.25)
        self.assertAlmostEqual(middle.mean().item(), 0.75)
        self.assertFalse(middle.requires_grad)

    def test_lora_starts_equivalent_and_trains_only_adapter(self):
        base = nn.Linear(4, 4)
        base_out = base(torch.ones(2, 4)).detach()
        layer = LoRALinear(base, rank=2)
        x = torch.ones(2, 4)
        self.assertTrue(torch.allclose(layer(x), base_out))
        layer(x).sum().backward()
        self.assertIsNone(layer.base.weight.grad)
        self.assertGreater(layer.b.grad.abs().sum().item(), 0)

    def test_cache_has_room_for_two_blocks(self):
        cache = make_kv_cache(2, 1, 192, 12, 128, torch.float16, 'cpu')
        self.assertEqual(len(cache), 2)
        self.assertEqual(tuple(cache[0]['k'].shape), (1, 192, 12, 128))
        self.assertEqual(cache[0]['global_end_index'].item(), 0)


if __name__ == '__main__':
    unittest.main()
