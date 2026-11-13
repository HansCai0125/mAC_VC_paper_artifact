from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import sqlite3
import zlib
import time
import numpy as np
from downstream.common import atomic_json, write_csv, sha256_file, digest_json, run_lock

PROJECT = Path(__file__).resolve().parents[3]
SAM = PROJECT / 'experiments/segmentation/results/main'
ANNOTATION = PROJECT.parent / 'Database/MS-COCO/annotations/instances_val2017.json'
NATURAL = PROJECT / 'results/collective_gain/runs'
EXACT = PROJECT / 'results/mask_polygon/runs'
MODELS = ('yolov8n', 'yolo11n', 'yolo11s')
VERSION = 'real-output-diagnostics-1.0.0'

@dataclass(frozen=True)
class Settings:
    theta: float = .25
    budget: int = 20
    bootstrap: int = 2000
    translations: int = 200
    cv_folds: int = 5
    cv_repeats: int = 10
    random_seed: int = 20261027
    gain_threshold: float = .05
    area_relative_epsilon: float = 1e-9
    min_stratum_images: int = 20
    min_stratum_targets: int = 60
    coarsening_deltas: tuple = (0., .25, .5)
    smoke: bool = False
    smoke_images: int = 12

def log(text):
    print(time.strftime('%H:%M:%S'), text, flush=True)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def unpack(blob):
    return json.loads(zlib.decompress(blob))

def seed_for(*values):
    return int(hashlib.sha256(':'.join(map(str, values)).encode()).hexdigest()[:16], 16)

def image_number(value):
    return int(Path(str(value)).stem)

def read_only_db(path):
    db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    return db

def file_identity(path):
    path = Path(path).resolve(); stat = path.stat()
    return {'path': str(path), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}

class Results:
    """Only this experiment's database is writable; one image is a transaction."""
    def __init__(self, directory):
        self.db = sqlite3.connect(Path(directory) / 'diagnostics.sqlite')
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS tasks(phase TEXT, run TEXT, image INTEGER, payload BLOB, PRIMARY KEY(phase,run,image))')
    def done(self, phase, run):
        return {r[0] for r in self.db.execute('SELECT image FROM tasks WHERE phase=? AND run=?',(phase,run))}
    def save(self, phase, run, image, payload):
        data=zlib.compress(json.dumps(payload, allow_nan=False, separators=(',',':')).encode(),3)
        with self.db:self.db.execute('INSERT INTO tasks VALUES(?,?,?,?)',(phase,run,int(image),data))
    def payloads(self, phase, run):
        for image, payload in self.db.execute('SELECT image,payload FROM tasks WHERE phase=? AND run=? ORDER BY image',(phase,run)):
            yield image,unpack(payload)
    def close(self):self.db.close()

def finite(value):
    return float(value) if value is not None and np.isfinite(value) else None
