import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from mpcawin_run import PAPER,evaluate_events,run_stream,resolve_config,load_configs

BASE=Path(__file__).resolve().parents[1]


class RunnerTests(unittest.TestCase):
    def test_import_does_not_execute_stream(self):
        for name in ['run_imprtotal.py','test_c_values.py']:
            spec=importlib.util.spec_from_file_location('import_only',BASE/name)
            module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
            self.assertTrue(callable(module.main))

    def test_remainder_is_explicit(self):
        x=pd.DataFrame(np.random.RandomState(2).normal(size=(11,2)))
        config=dict(PAPER,batch_size=4,min_window_size=1,max_window_size=1)
        with self.assertRaises(ValueError): run_stream(x,config)
        table,summary=run_stream(x,dict(config,remainder='drop'))
        self.assertEqual(len(table),2); self.assertEqual(summary['dropped_rows'],3)
        table,summary=run_stream(x,dict(config,remainder='process'))
        self.assertEqual(len(table),3); self.assertEqual(table.iloc[-1].end_row_exclusive,11)

    def test_horizon_and_batch_numbering(self):
        result=evaluate_events([3,12,40],[80],10,5)
        self.assertEqual(result['matches'][0]['detection_batch_id0'],12)
        self.assertEqual(result['matches'][0]['index_delay_batches'],4)
        self.assertEqual(result['matches'][0]['consumed_new_concept_batches'],5)
        self.assertEqual(result['unmatched_detection_batches'],[3,40])
        self.assertEqual(result['detections_before_first_boundary'],[3])
        self.assertTrue(evaluate_events([40],[80],10,5)['matches'][0]['missed'])
        result=evaluate_events([12],[80,120],10,20)
        self.assertTrue(result['matches'][0]['missed'])
        self.assertEqual(result['matches'][1]['detection_batch_id0'],12)
        result=evaluate_events([2],[8],4,5,processed_rows=11)
        self.assertEqual(result['matches'][0]['consumed_new_concept_records'],3)

    def test_config_validation(self):
        with self.assertRaises(ValueError): resolve_config(dict(typo=1))
        with self.assertRaises(ValueError): resolve_config(dict(name='../bad'))
        self.assertEqual(load_configs(BASE/'configs/paper.json')[0]['coef_std'],2)
        self.assertEqual(len(load_configs(BASE/'configs/experiments.json')),6)

    def test_import_from_outside_project(self):
        with tempfile.TemporaryDirectory() as directory:
            completed=subprocess.run([sys.executable,'-c','import mpcawin; from mpcawin_run import make_detector; print(mpcawin.__file__)'],
                                     cwd=directory,capture_output=True,text=True,check=True)
        self.assertEqual((BASE/'src/mpcawin.py').resolve(),Path(completed.stdout.strip()).resolve())
