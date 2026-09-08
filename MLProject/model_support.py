"""Shared data contract and evaluation semantics; no model selection here."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mlflow
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'sqli_xss_preprocessing'
EVIDENCE = ROOT / 'evidence'


def configure():
    EVIDENCE.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_URI', (ROOT/'mlruns').as_uri()))
    mlflow.set_experiment('MSML_DaudHidayatRamadhan')


def load_data():
    manifest = json.loads((DATA/'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Processed checksum mismatch: {name}')
    train, test = [pd.read_csv(DATA/f'{name}.csv.gz') for name in ('train','test')]
    if not set(train.sample_id).isdisjoint(test.sample_id):
        raise ValueError('Train/test identity overlap')
    return train, test, manifest


def make_model(c=1.0):
    return Pipeline([
        ('features', ColumnTransformer([('text', TfidfVectorizer(analyzer='char',
            ngram_range=(3,4),max_features=20000,min_df=2,dtype=np.float32,
            lowercase=False), 'Sentence')], sparse_threshold=1.0)),
        ('classifier', LinearSVC(C=c,dual='auto',max_iter=5000,random_state=42)),
    ])


def scores(truth, prediction, prefix, training=False):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    if training:
        return {'training_accuracy_score': float(accuracy_score(truth,prediction)),
                'training_score': float(accuracy_score(truth,prediction)),
                'training_precision_score': float(precision_score(truth,prediction,average='weighted',zero_division=0)),
                'training_recall_score': float(recall_score(truth,prediction,average='weighted',zero_division=0)),
                'training_f1_score': float(f1_score(truth,prediction,average='weighted',zero_division=0))}
    return {prefix+'accuracy': float(accuracy_score(truth,prediction)),
            prefix+'macro_precision': float(precision_score(truth,prediction,average='macro',zero_division=0)),
            prefix+'macro_recall': float(recall_score(truth,prediction,average='macro',zero_division=0)),
            prefix+'macro_f1': float(f1_score(truth,prediction,average='macro',zero_division=0)),
            prefix+'weighted_f1': float(f1_score(truth,prediction,average='weighted',zero_division=0))}


def artifact_paths(run_id):
    client = mlflow.tracking.MlflowClient()
    def walk(prefix=''):
        result=[]
        for item in client.list_artifacts(run_id,prefix):
            result.extend(walk(item.path) if item.is_dir else [item.path])
        return result
    return sorted(walk())


def report_model(model,test):
    from sklearn.metrics import classification_report, ConfusionMatrixDisplay
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pred=model.predict(test[["Sentence"]])
    report=classification_report(test.target,pred,output_dict=True,zero_division=0)
    (EVIDENCE/'classification_report.json').write_text(json.dumps(report,indent=2)+'\n')
    ConfusionMatrixDisplay.from_predictions(test.target,pred,xticks_rotation=25)
    plt.tight_layout();plt.savefig(EVIDENCE/'holdout_confusion_matrix.png',dpi=140);plt.close()
    return pred
