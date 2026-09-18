"""Local review app: python -m uvicorn app.review:app --host 127.0.0.1 --port 8101."""
import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'ml/data/raw/2026_09_16'
DERIVED = ROOT / 'ml/data/derived/2026_09_16'
app = FastAPI(title='Equipment review (local only)')


def assets():
    with sqlite3.connect(f'{(DERIVED / "catalog.sqlite").as_uri()}?mode=ro', uri=True) as db:
        rows = db.execute('SELECT path, sha256, metadata FROM media WHERE kind=? ORDER BY path', ('image',)).fetchall()
    items = {}
    for path, sha, meta in rows:
        if json.loads(meta)['status'] == 'metadata_ok':
            items.setdefault(sha, {'id': sha, 'path': RAW / path, 'source': path, 'kind': 'image'})
    for frame in json.loads((DERIVED / 'frames/manifest.json').read_text(encoding='utf-8')):
        if frame['status'] == 'extracted':
            items.setdefault(frame['sha256'], {'id': frame['sha256'], 'path': DERIVED / 'frames' / frame['path'],
                             'source': frame['source'], 'frame': frame['actual_frame'], 'kind': 'frame'})
    return items


ASSETS = assets()


def connect():
    db = sqlite3.connect(DERIVED / 'reviews.sqlite')
    db.execute('CREATE TABLE IF NOT EXISTS reviews (asset_id TEXT PRIMARY KEY, payload TEXT NOT NULL)')
    return db


class Region(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    group: str = Field(default='', max_length=100)
    label: str = Field(default='', max_length=100)
    reviewer: str = Field(default='', max_length=100)
    evidence: str = Field(default='', max_length=1000)
    confirmed: bool = False


class Review(BaseModel):
    regions: list[Region] = Field(max_length=100)
    unusable: bool = False


def item(key):
    if key not in ASSETS:
        raise HTTPException(404, 'Unknown asset')
    return ASSETS[key]


@app.get('/')
def home():
    return FileResponse(ROOT / 'app/static/review.html')


@app.get('/api/assets')
def listing(offset: int = 0, limit: int = 40, kind: str = ''):
    rows = [v for v in ASSETS.values() if not kind or v['kind'] == kind]
    return {'total': len(rows), 'items': [{k: v for k, v in r.items() if k != 'path'}
                                        for r in rows[max(0, offset):max(0, offset) + min(100, max(1, limit))]]}


@app.get('/api/assets/{key}/image')
def media(key: str):
    return FileResponse(item(key)['path'])


@app.get('/api/assets/{key}/review')
def read_review(key: str):
    item(key)
    with connect() as db:
        row = db.execute('SELECT payload FROM reviews WHERE asset_id=?', (key,)).fetchone()
    return json.loads(row[0]) if row else {'regions': [], 'unusable': False}


@app.put('/api/assets/{key}/review')
def save_review(key: str, review: Review):
    with Image.open(item(key)['path']) as image:
        width, height = image.size
    for region in review.regions:
        if region.x + region.width > width or region.y + region.height > height:
            raise HTTPException(422, 'Region outside image')
        if region.confirmed and not all(s.strip() for s in (region.label, region.reviewer, region.evidence)):
            raise HTTPException(422, 'Confirmed labels require name, reviewer and evidence')
    with connect() as db:
        db.execute('INSERT OR REPLACE INTO reviews VALUES (?,?)', (key, review.model_dump_json()))
    return {'saved': True}
