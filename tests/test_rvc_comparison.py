import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from core.rvc_comparison import PRESETS, listening_target, match_loudness
from core.errors import VoiceCloneError


class ComparisonTests(unittest.TestCase):
    def test_each_variant_changes_only_one_parameter(self):
        baseline = PRESETS[0]
        self.assertEqual(baseline[1:3], (0.75, 0.33))
        for preset in PRESETS[1:]:
            self.assertEqual(sum(x != y for x, y in zip(baseline[1:3], preset[1:3])), 1)

    def test_common_target_preserves_true_peak_headroom(self):
        measurements = [{'lufs': -12, 'true_peak_dbtp': -1}, {'lufs': -30, 'true_peak_dbtp': -1}]
        target = listening_target(measurements)
        self.assertEqual(target, -31)
        for m in measurements:
            self.assertLessEqual(m['true_peak_dbtp'] + target - m['lufs'], -2)

    def test_gain_matching_keeps_timing_and_dynamics(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / 'source.wav', Path(temp) / 'matched.wav'
            samples = np.array([[0.1, 0.2], [0.2, 0.4], [-0.1, -0.2]], dtype='float32')
            sf.write(source, samples, 48000, subtype='PCM_24')
            gain = match_loudness(source, destination, {'lufs': -17}, -23)
            result, rate = sf.read(destination, always_2d=True)
            self.assertEqual(gain, -6)
            self.assertEqual(rate, 48000)
            self.assertEqual(result.shape, samples.shape)
            np.testing.assert_allclose(result, samples * 10 ** (-6 / 20), atol=2e-7)
            self.assertEqual(sf.info(destination).subtype, 'PCM_24')

    def test_clipping_is_refused_instead_of_limited(self):
        with tempfile.TemporaryDirectory() as temp:
            source, destination = Path(temp) / 'source.wav', Path(temp) / 'matched.wav'
            sf.write(source, np.array([0.8, -0.8]), 48000)
            with self.assertRaises(VoiceCloneError):
                match_loudness(source, destination, {'lufs': -30}, -10)
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
