import importlib.util
import unittest
from pathlib import Path
import numpy as np
from kswin_compat import KSWIN,RiverKSWINAdapter

BASE=Path(__file__).resolve().parents[1]


class KSWINTests(unittest.TestCase):
    def test_legacy_decisions_and_windows_match_saved_implementation(self):
        spec=importlib.util.spec_from_file_location('old_ks',BASE/'baselines/before_repair/src/kswin_compat.py')
        old=importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
        stream=np.r_[np.zeros(45),[1,0,0,1,0],np.ones(60),np.zeros(50)]
        # Global seeded RNG is used only by the archived reference.
        self.addCleanup(np.random.set_state, np.random.get_state())
        np.random.seed(20)
        legacy=old.KSWIN(alpha=.009,window_size=20,stat_size=5)
        repaired=KSWIN(alpha=.009,window_size=20,stat_size=5,random_state=20)
        for index,value in enumerate(stream):
            if index==90: legacy.reset(); repaired.reset()
            legacy.add_element(value); repaired.add_element(value)
            self.assertEqual(legacy.detected_change(),repaired.detected_change())
            np.testing.assert_array_equal(legacy.window,repaired.window)
            self.assertEqual(legacy.p_value,repaired.p_value)

    def test_reproducibility_and_independence(self):
        a=KSWIN(window_size=20,stat_size=5,random_state=4)
        b=KSWIN(window_size=20,stat_size=5,random_state=4)
        distraction=KSWIN(window_size=20,stat_size=5,random_state=99)
        for value in [0]*30+[1]*30:
            a.add_element(value)
            distraction.add_element(1-value)
            b.add_element(value)
            self.assertEqual(a.last_result,b.last_result)

    def test_validation_and_initial_window_copy(self):
        for kwargs in [dict(alpha=0),dict(alpha=1),dict(window_size=0),dict(stat_size=0),dict(window_size=5,stat_size=4)]:
            with self.assertRaises((ValueError,TypeError)): KSWIN(**kwargs)
        with self.assertRaises(ValueError): RiverKSWINAdapter(window_size=9,stat_size=5)
        data=np.array([0.,1.]); detector=KSWIN(data=data)
        data[0]=99
        self.assertEqual(detector.window[0],0)
        before=detector.window.copy()
        for invalid in [np.nan, np.inf, '1', 1+2j, [1]]:
            with self.assertRaises(ValueError): detector.add_element(invalid)
            np.testing.assert_array_equal(detector.window,before)
            self.assertEqual(detector.n,0)

    def test_snapshot_and_exact_separation(self):
        d=KSWIN(alpha=.009,window_size=20,stat_size=5,random_state=20)
        detections=[]
        for i,value in enumerate([0]*40+[1]*40):
            d.add_element(value)
            if d.detected_change(): detections.append(i)
        self.assertTrue(detections)
        d.reset(); self.assertIsNone(d.last_result)
