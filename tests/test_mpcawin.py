import copy
import unittest
import numpy as np
from mpcawin import MPCAWIN


class RecordingDetector:
    def reset(self): self.last_result=None
    def add_element(self,value):
        self.last_result=dict(p_value=.005,statistic=1.,tested=True,drift_detected=True)
    def detected_change(self): return True


class MPCAWINTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.RandomState(7).normal(size=(100,3))

    def test_instance_ownership_and_global_rng(self):
        before=np.random.get_state()
        first,second=MPCAWIN(),MPCAWIN()
        self.assertIsNot(first.detector,second.detector)
        first.detector.add_element(1)
        self.assertEqual(len(second.detector.window),0)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        self.assertEqual(before[2:],after[2:])

    def test_reset_capacity_and_first_full_alert_window(self):
        model=MPCAWIN(min_window_size=2,max_window_size=2,detect_strategy='alert_rate',alert_window_size=2,alpha=1)
        model.set_concept(self.x)
        model.add_element_and_detected_change(self.x)
        model.add_element_and_detected_change(self.x)
        changed=self.x.copy(); changed[:,0]*=20
        self.assertFalse(model.add_element_and_detected_change(changed))
        self.assertTrue(model.add_element_and_detected_change(changed))
        self.assertEqual(model.alert_window.maxsize,2)
        self.assertEqual(model.concept_id,1)
        self.assertIsNone(model.threshold)
        self.assertEqual(model.last_result['alert'],1)
        self.assertEqual(model.last_result['concept_before'],0)
        self.assertEqual(model.last_result['concept_after'],1)
        model.reset()
        self.assertIsNone(model.last_result)
        self.assertIsNone(model.last_angle)

    def test_invalid_input_preserves_concept_and_decision(self):
        model=MPCAWIN(); model.set_concept(self.x)
        model.add_element_and_detected_change(self.x)
        eigen=model.actual_concept_eigenvalues.copy()
        decision=copy.deepcopy(model.last_result)
        for bad in [np.full((100,3),np.nan),np.ones((100,3)),np.ones((1,3))]:
            with self.assertRaises(ValueError): model.set_concept(bad)
            np.testing.assert_array_equal(eigen,model.actual_concept_eigenvalues)
            self.assertEqual(decision,model.last_result)
            with self.assertRaises(ValueError): model.add_element_and_detected_change(bad)
            self.assertEqual(decision,model.last_result)

    def test_schema_and_component_validation(self):
        import pandas as pd
        x=pd.DataFrame(self.x,columns=['a','b','c'])
        model=MPCAWIN(); model.set_concept(x)
        with self.assertRaises(ValueError): model.add_element_and_detected_change(x[['c','b','a']])
        with self.assertRaises(ValueError): model.add_element_and_detected_change(x.iloc[:2])
        with self.assertRaises(ValueError): model.add_element_and_detected_change(x.astype(str))

    def test_snapshot_survives_detection_reset(self):
        detector=RecordingDetector()
        model=MPCAWIN(min_window_size=1,max_window_size=1,detector=detector)
        model.set_concept(self.x)
        model.add_element_and_detected_change(self.x)
        self.assertTrue(model.add_element_and_detected_change(self.x))
        self.assertIsNone(detector.last_result)
        self.assertEqual(model.last_result['ks']['p_value'],.005)

    def test_reference_pca_is_not_refitted(self):
        model=MPCAWIN(); model.set_concept(self.x)
        components=model.pca.components_.copy()
        changed=self.x.copy(); changed[:,0]*=3
        model.add_element_and_detected_change(changed)
        np.testing.assert_array_equal(components,model.pca.components_)

    def test_validation(self):
        for kwargs in [dict(min_window_size=0),dict(min_window_size=-1),dict(max_window_size=1),
                       dict(coef_std=-1),dict(coef_std=np.inf),dict(alpha=0),dict(random_seed='None')]:
            with self.assertRaises((ValueError,TypeError)): MPCAWIN(**kwargs)
        with self.assertRaises(TypeError): MPCAWIN(min_window_size=1.5)
        MPCAWIN(random_seed=None)

    def test_two_drift_cycles_and_isolated_outlier(self):
        model=MPCAWIN(min_window_size=2,max_window_size=2,detect_strategy='alert_rate',alert_window_size=3,alpha=1)
        model.set_concept(self.x)
        for _ in range(2): model.add_element_and_detected_change(self.x)
        changed=self.x.copy(); changed[:,0]*=20
        self.assertFalse(model.add_element_and_detected_change(changed))
        for _ in range(3): self.assertFalse(model.add_element_and_detected_change(self.x))
        events=[model.add_element_and_detected_change(changed) for _ in range(3)]
        self.assertEqual(events,[False,False,True])
        for _ in range(2): self.assertFalse(model.add_element_and_detected_change(changed))
        events=[model.add_element_and_detected_change(self.x) for _ in range(3)]
        self.assertEqual(events,[False,False,True])
        self.assertEqual(model.concept_id,2)
