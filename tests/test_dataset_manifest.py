import importlib.util
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from dataset_manifest import describe_concatenation,write_manifest,read_manifest,select_features

DATA_SCRIPTS=Path(__file__).resolve().parents[1]/'dataset_tools'


def load_script(name):
    spec=importlib.util.spec_from_file_location(name,DATA_SCRIPTS/name)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class ManifestTests(unittest.TestCase):
    def test_order_hash_features_and_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            first=pd.DataFrame(dict(x=[1.,2.,3.],y=[2.,4.,3.],attack_flag=[0,1,0],attack_step=[0,7,0]))
            second=first+1
            first.to_csv(path/'a.csv',index=False); second.to_csv(path/'b.csv',index=False)
            pd.concat([first,second],ignore_index=True).to_csv(path/'total.csv',index=False)
            manifest=describe_concatenation(path/'total.csv',[path/'a.csv',path/'b.csv'])
            self.assertEqual(manifest['boundaries'][0]['row'],3)
            self.assertEqual(manifest['feature_columns'],['x','y'])
            write_manifest(path/'total.csv',manifest)
            self.assertEqual(read_manifest(path/'total.csv')['data']['rows'],6)
            pd.concat([second,first]).to_csv(path/'wrong.csv',index=False)
            with self.assertRaises(ValueError): describe_concatenation(path/'wrong.csv',[path/'a.csv',path/'b.csv'])
            with (path/'total.csv').open('a') as out: out.write('9,9,0,0\n')
            with self.assertRaises(ValueError): read_manifest(path/'total.csv')
        with self.assertRaises(ValueError): select_features(first,['attack_flag'])

    def test_script_imports_and_mixed_partial_batches(self):
        split=load_script('split_batches.py')
        for name in ['prepare_impr_stratified.py','create_imprtotal.py','compare_columns.py']:
            self.assertTrue(callable(load_script(name).main))
        frame=pd.DataFrame(dict(attack_flag=[0]*11))
        metadata=dict(sources=[dict(name='a',start_row=0,end_row_exclusive=5),
                               dict(name='b',start_row=5,end_row_exclusive=11)])
        table=split.summarize_batches(frame,metadata,4)
        self.assertEqual(table.iloc[1].source,'Mixed')
        self.assertTrue(table.iloc[2].partial)
        self.assertEqual(table.iloc[2].rows,3)

    def test_sampling_provenance_and_no_overwrite(self):
        generator=load_script('prepare_impr_stratified.py')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            rng=np.random.RandomState(3)
            frame=pd.DataFrame(dict(x=rng.normal(size=100),y=rng.normal(size=100),
                                    attack_flag=[0]*80+[1]*20,attack_name=['Benign']*80+['A']*10+['B']*10,
                                    attack_step=[0]*80+[7]*20))
            frame.to_csv(path/'source.csv',index=False)
            kwargs=dict(target_size=30,attack_ratio=.2,min_per_family=1,seed=42,chunk_size=17)
            with contextlib.redirect_stdout(io.StringIO()):
                generator.create_stratified_impr(path/'source.csv',path/'sample.csv',path/'summary.csv','test',**kwargs)
                generator.create_stratified_impr(path/'source.csv',path/'repeat.csv',path/'repeat_summary.csv','test',**kwargs)
            self.assertEqual((path/'sample.csv').read_bytes(),(path/'repeat.csv').read_bytes())
            self.assertEqual((path/'sample.source_rows.csv').read_bytes(),(path/'repeat.source_rows.csv').read_bytes())
            result=pd.read_csv(path/'sample.csv')
            indices=pd.read_csv(path/'sample.source_rows.csv').source_row
            pd.testing.assert_frame_equal(result,frame.iloc[indices].reset_index(drop=True))
            metadata=read_manifest(path/'sample.csv')
            self.assertEqual(metadata['sampling']['seed'],42)
            self.assertEqual(int(result.attack_flag.sum()),6)
            with self.assertRaises(FileExistsError):
                generator.create_stratified_impr(path/'source.csv',path/'sample.csv',path/'summary.csv','test',**kwargs)
