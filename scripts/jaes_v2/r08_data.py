"""Bounded access to the pinned author ZIP and deterministic audio eligibility."""
import csv
from collections import Counter, defaultdict
from hashlib import sha256
import io
from pathlib import PurePosixPath
import re
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED

import numpy as np
import soundfile as sf

from src.lava.jaes_v2.controls import prepare_headroom
from src.lava.jaes_v2.preprocessing import Preprocessor
from src.lava.jaes_v2.transforms import load_audio

REVISION = 'eee168f92c367f8c82ff2cf42b6f61e362fd6211'
RELEASE = 'mueller91/In-The-Wild'
URL = f'https://huggingface.co/datasets/{RELEASE}/resolve/{REVISION}/release_in_the_wild.zip'
ETAG = '"ec530d2076f973d40a8e388e559888f1b4fea467db8126c811d34dd508ee106f"'
ZIP_BYTES = 8161489023
ZIP_SHA256 = '46665f30a6758b45642c659c7e0e80da2ba3d121b064716824a56a010ddf9e1a'
META_SHA256 = '2e404150eff3cf171b766da81013a93248cbb219682fff420bbadb6b86a18795'
PREFIX = 'release_in_the_wild/'
META_BUDGET = 12 * 1024**2
MAX_MEMBER = 32 * 1024**2
SEED = 20261112


class RemoteZip(io.RawIOBase):
    """Provide zipfile with strict, bounded random-access reads of one immutable object."""

    def __init__(self, budget, guard, *, url=URL, length=ZIP_BYTES, etag=ETAG, debit=None):
        self.url, self.length, self.etag = url, length, etag
        self.budget, self.guard = budget, guard
        self.debit = debit
        self.pos = self.transferred = self.requested = 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        pos = {0: 0, 1: self.pos, 2: self.length}[whence] + offset
        if pos < 0:
            raise ValueError('Negative archive offset')
        self.pos = pos
        return pos

    def read(self, size=-1):
        start = self.pos
        end = self.length if size < 0 else min(self.length, start + size)
        if end <= start:
            return b''
        count = end - start
        if count > MAX_MEMBER or self.requested + count > self.budget:
            raise ValueError('Range byte budget exceeded')
        self.guard()
        if self.debit:
            self.debit(count)
        self.requested += count
        request = Request(self.url, headers={'Range': f'bytes={start}-{end-1}', 'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=30) as response:
            if (response.status != 206
                    or response.headers.get('Content-Range') != f'bytes {start}-{end-1}/{self.length}'
                    or response.headers.get('ETag') != self.etag
                    or response.headers.get('Content-Length') not in (None, str(count))):
                raise ValueError('Archive identity or range response mismatch')
            data = response.read(count)
            self.transferred += len(data)
            if len(data) != count:
                raise ValueError('Unexpected range body size')
            self.url = response.geturl()
        self.pos = end
        return data


def member_name(value):
    path = PurePosixPath(value)
    if (not re.fullmatch(r'[\w.-]+\.wav', value, re.I | re.ASCII)
            or len(path.parts) != 1 or path.name != value):
        raise ValueError(f'Unsafe WAV member: {value!r}')
    return PREFIX + value


def parse_metadata(payload):
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    if set(reader.fieldnames or ()) != {'file', 'speaker', 'label'} or len(reader.fieldnames) != 3:
        raise ValueError('Unexpected pinned metadata schema')
    rows = {}
    for row in reader:
        if set(row) != {'file', 'speaker', 'label'} or not all(isinstance(v, str) and v.strip() == v and v for v in row.values()):
            raise ValueError('Malformed metadata row')
        name = member_name(row['file'])
        if name.casefold() in rows or row['label'] not in ('bonafide', 'bona-fide', 'spoof'):
            raise ValueError('Duplicate filename or unexpected label')
        rows[name.casefold()] = dict(member=name, sample_id=row['file'], speaker_id=row['speaker'],
                                     label=int(row['label'] == 'spoof'), source_group=None)
    return list(rows.values())


def inventory(archive):
    entries = archive.infolist()
    if len({m.filename.casefold() for m in entries}) != len(entries):
        raise ValueError('Duplicate archive members')
    meta = archive.getinfo(PREFIX + 'meta.csv')
    if meta.file_size > META_BUDGET:
        raise ValueError('Oversized metadata')
    payload = archive.read(meta)
    if sha256(payload).hexdigest() != META_SHA256:
        raise ValueError('Metadata checksum mismatch')
    rows = parse_metadata(payload)
    members = {m.filename: m for m in entries if m.filename.lower().endswith('.wav')}
    if set(members) != {r['member'] for r in rows} or len(rows) != 31779:
        raise ValueError('Pinned archive and metadata identities differ')
    for m in members.values():
        if m.header_offset < 0 or m.header_offset + m.compress_size + 30 > archive.start_dir:
            raise ValueError('WAV member offset outside ZIP data')
    return rows, members


def speaker_key(value):
    return ''.join(re.findall(r'[^\W_]+', value.casefold()))


def shortlist(rows, speakers, per_class, blocked_ids=(), blocked_speakers=(), aliases=None):
    groups = defaultdict(lambda: defaultdict(list))
    aliases = aliases or {}
    key = lambda value: sha256(f'{SEED}:{value}'.encode()).hexdigest()
    for row in rows:
        canonical = speaker_key(row['speaker_id'])
        canonical = aliases.get(canonical, canonical)
        if (not canonical or PurePosixPath(row['sample_id']).stem.casefold() in blocked_ids
                or row['sample_id'].casefold() in blocked_ids or canonical in blocked_speakers):
            continue
        groups[canonical][row['label']].append(dict(row, speaker_id=canonical, speaker_label=row['speaker_id']))
    support = {s: [len(v[0]), len(v[1])] for s, v in groups.items()}
    eligible = sorted((s for s, v in support.items() if min(v) >= per_class), key=key)
    if len(eligible) < speakers:
        raise ValueError(f'Only {len(eligible)} speaker groups meet the {per_class}-per-class metadata quota')
    chosen = [row for s in eligible[:speakers] for y in (0, 1)
              for row in sorted(groups[s][y], key=lambda r: key(r['sample_id']))[:per_class]]
    return sorted(chosen, key=lambda r: r['sample_id']), support


def read_member(archive, row):
    name = member_name(row['sample_id'])
    if name != row['member']:
        raise ValueError('ZIP member identity mismatch')
    member = archive.getinfo(name)
    if (not 0 < member.file_size <= MAX_MEMBER or member.compress_size > MAX_MEMBER
            or member.compress_type not in (ZIP_STORED, ZIP_DEFLATED) or member.flag_bits & 1):
        raise ValueError(f'Unsupported member: {name}')
    return archive.read(member)  # zipfile checks the decompressed CRC and local member name.


def qualified_audio(payload):
    """Audit the entire WAV. Preserve the registered first-three-seconds input policy."""
    digest = sha256()
    with sf.SoundFile(io.BytesIO(payload)) as f:
        if f.format not in ('WAV', 'WAVEX') or not f.frames or not 1 <= f.channels <= 8 or not 8000 <= f.samplerate <= 192000:
            raise ValueError('Unsupported WAV format')
        header = dict(sample_rate=f.samplerate, channels=f.channels, frames=f.frames,
                      duration=f.frames / f.samplerate, encoding=f.subtype)
        digest.update(f'{f.samplerate}:{f.channels}:'.encode())
        count = 0
        for block in f.blocks(blocksize=65536, always_2d=True, dtype='float32'):
            if not np.isfinite(block).all():
                raise ValueError('Nonfinite decoded samples')
            count += len(block)
            digest.update(block.astype('<f4', copy=False).tobytes())
        if count != f.frames:
            raise ValueError('Truncated WAV')
    x = clean_input(io.BytesIO(payload))
    return dict(header, sha256=sha256(payload).hexdigest(), decoded_sha256=digest.hexdigest(),
                shape=list(x.shape), bytes=len(payload))


def clean_input(path):
    audio = load_audio(path, sample_rate=22050, max_seconds=3)
    if not audio.samples.size or not np.isfinite(audio.samples).all() or not np.any(audio.samples):
        raise ValueError('Nonfinite, empty or silent observed waveform')
    waveform, _ = prepare_headroom(audio, .25)
    x = Preprocessor('P0').extract(waveform.samples)
    if x.ndim != 4 or not np.isfinite(x).all():
        raise ValueError('Invalid P0 tensor')
    return x
