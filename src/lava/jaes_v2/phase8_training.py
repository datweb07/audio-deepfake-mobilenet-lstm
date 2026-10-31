"""Versioned CPU/GPU-portable trainer. Checkpoints commit at optimizer boundaries.

Stateless dropout, epoch-derived sampling and explicit cursors replace implicit
Keras fit iterator state. Production and Phase1--7 code are not modified.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import Counter, OrderedDict
from time import perf_counter
import json, os, re, shutil, sys, platform
import numpy as np
from .evidence import file_hash, object_hash, read_csv, write_csv, write_json
from .phase7_data import safe_path
from .phase8_objectives import group_key, dro_update, bce, weighted_sum

BASE = Path('outputs/jaes_v2/phase8')
MEDIUM = 'outputs/jaes_v2/phase7/selections/medium_all_attack_v1/manifest.csv'
FOUNDATION = 'outputs/jaes_v2/datasets/asv2019_foundation_v1/train_manifest.csv'
DEV = ('clean', 'gain_half', 'gain_quarter', 'noise5', 'opus64', 'A_then_B', 'B_then_A')
BANKS = {'clean': ('clean',), 'channel': ('clean', 'gain_half', 'noise10', 'opus24'),
         'rawboost_finite': ('clean', 'rawboost0', 'rawboost1')}


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    research_phase: int
    dataset_version: str
    source_manifest: str
    manifest_sha256: str
    development_manifest: str
    development_sha256: str
    training_scope: str
    architecture: str
    pretrained_weights: str
    preprocessing_policy: str
    augmentation_policy: str
    training_objective: str
    objective_parameters: dict
    trainable_backbone_policy: str
    BatchNorm_policy: str
    training_seed: int
    augmentation_seed: int
    sampling: str
    learning_rate: float
    finetune_learning_rate: float
    scheduler: str
    batch_size: int
    microbatch_size: int
    gradient_accumulation: int
    warmup_epochs: int
    finetune_epochs: int
    maximum_updates: int
    checkpoint_policy: str
    checkpoint_every_updates: int
    validation_policy: str
    early_stopping_patience: int
    clean_REAL_FA_cap: float
    clean_FAKE_miss_cap: float
    resource_limits: dict
    runtime: str
    protocol_path: str
    tiny_per_attack: int

    def __post_init__(self):
        if not re.fullmatch(r'p8_[a-z0-9_]+', self.experiment_id) or self.research_phase != 8:
            raise ValueError('Phase8 identity required')
        if self.dataset_version != 'ASVspoof2019_LA' or self.source_manifest not in (MEDIUM, FOUNDATION):
            raise ValueError('Only locked allowed train/development manifests')
        expected_dev = MEDIUM if self.source_manifest == MEDIUM else FOUNDATION.replace('train_manifest', 'development_manifest')
        if self.development_manifest != expected_dev:
            raise ValueError('Reserved/foreign development role forbidden')
        for h in (self.manifest_sha256, self.development_sha256):
            if not re.fullmatch('[a-f0-9]{64}', h): raise ValueError('Manifest SHA256 required')
        if self.training_scope not in ('TINY_LIFECYCLE', 'DEVELOPMENT_RESEARCH') or self.architecture != 'MobileNetV3Small-LSTM128' or self.pretrained_weights != 'imagenet_fresh_head' or self.preprocessing_policy != 'P0':
            raise ValueError('Architecture/scope contract')
        if self.augmentation_policy not in BANKS or self.training_objective not in ('ERM', 'WEIGHTED_BCE', 'GROUP_DRO', 'EQUAL_VIEW', 'CONSISTENCY'):
            raise ValueError('Only reviewed standard baselines; M3 rejected')
        if self.augmentation_seed!=5804:raise ValueError('Inherited finite waveform bank requires seed5804')
        if set(self.objective_parameters) != {'class_weights', 'dro_rate', 'dro_floor', 'consistency_weight'}:
            raise ValueError('Objective schema')
        p = self.objective_parameters
        if len(p['class_weights']) != 2 or any(not np.isfinite(v) or v <= 0 for v in p['class_weights']) or not 0 < p['dro_rate'] <= 1 or not 0 < p['dro_floor'] < 1 or not 0 <= p['consistency_weight'] <= 1:
            raise ValueError('Objective values')
        if self.trainable_backbone_policy not in ('HEAD_ONLY', 'WARMUP_PARTIAL') or self.BatchNorm_policy != 'ALL_FROZEN': raise ValueError('Trainability contract')
        if self.sampling not in ('natural', 'class_balanced', 'group_balanced'): raise ValueError('Sampling protocol')
        if (self.training_objective == 'GROUP_DRO') != (self.sampling == 'group_balanced'):
            raise ValueError('DRO requires complete class/attack-balanced update groups')
        if self.training_objective == 'GROUP_DRO' and self.batch_size < 7: raise ValueError('DRO needs seven supported source groups per update')
        if self.batch_size != self.microbatch_size * self.gradient_accumulation or not 1 <= self.microbatch_size <= 16 or self.batch_size > 128: raise ValueError('Effective batch contract')
        if not 1 <= self.warmup_epochs <= 100 or not 0 <= self.finetune_epochs <= 100 or (self.trainable_backbone_policy == 'HEAD_ONLY' and self.finetune_epochs): raise ValueError('Stage epochs')
        if self.trainable_backbone_policy == 'WARMUP_PARTIAL' and not self.finetune_epochs: raise ValueError('Fine-tune stage missing')
        if not 0 < self.learning_rate <= .01 or not 0 < self.finetune_learning_rate <= .001 or self.scheduler != 'linear_half_per_stage': raise ValueError('Optimizer contract')
        if self.maximum_updates <= 0 or self.checkpoint_every_updates < 1 or self.early_stopping_patience < 1 or self.runtime not in ('cpu', 'gpu') or self.validation_policy != 'A_EVERY_EPOCH_ALL_CONDITIONS_FINAL_ALL_DEV': raise ValueError('Lifecycle contract')
        if self.checkpoint_policy not in ('CLEAN_BCE', 'CLEAN_RISK_CONSTRAINED_ROBUSTNESS') or not all(0 <= v <= 1 for v in (self.clean_REAL_FA_cap, self.clean_FAKE_miss_cap)): raise ValueError('Selection contract')
        if set(self.resource_limits) != {'seconds_per_invocation', 'disk_reserve_gib', 'max_process_mib', 'tensor_lru_items'} or self.resource_limits['disk_reserve_gib'] < 15 or self.resource_limits['seconds_per_invocation'] <= 0 or self.resource_limits['max_process_mib'] <= 0 or not 0 <= self.resource_limits['tensor_lru_items'] <= 8: raise ValueError('Resource contract')
        if (self.training_scope == 'TINY_LIFECYCLE') != (self.tiny_per_attack > 0): raise ValueError('Explicit tiny scope required')
        if self.trainable_backbone_policy == 'HEAD_ONLY' and self.source_manifest == FOUNDATION: raise ValueError('Full-foundation frozen cache not prepared; use partial policy')


def initialize(runtime):
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
    if runtime == 'cpu': os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import tensorflow as tf
    try:
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)
    except RuntimeError: pass
    if runtime == 'cpu': tf.config.set_visible_devices([], 'GPU')
    if runtime == 'gpu':
        devices = tf.config.list_physical_devices('GPU')
        if not devices: raise RuntimeError('GPU_RUNTIME_NOT_VERIFIED: no TensorFlow GPU')
        for d in devices: tf.config.experimental.set_memory_growth(d, True)
    tf.config.experimental.enable_op_determinism()
    return tf


def code_contract(root):
    files = ('src/lava/jaes_v2/phase8_training.py', 'src/lava/jaes_v2/phase8_objectives.py',
             'src/lava/jaes_v2/phase7_data.py', 'src/lava/jaes_v2/c2_data.py',
             'src/lava/jaes_v2/preprocessing.py', 'src/lava/jaes_v2/transforms.py', 'src/lava/jaes_v2/controls.py',
             'src/lava/jaes_v2/rawboost.py','src/lava/jaes_v2/vendor/rawboost/RawBoost.py',
             'src/preprocessing.py', 'src/model.py', 'config.py')
    result={f: file_hash(root / f) for f in files}
    executable=shutil.which('ffmpeg')
    if not executable:raise ValueError('ffmpeg required')
    result['ffmpeg_binary']=file_hash(Path(executable))
    return result


def source_path(root, declared):
    """Allow sealed absolute Windows dataset paths only inside this workspace."""
    from pathlib import PureWindowsPath
    path=Path(declared)
    if os.name!='nt' and PureWindowsPath(declared).is_absolute():
        parts=PureWindowsPath(declared).parts
        if root.name not in parts:raise ValueError('Foreign Windows dataset root')
        path=root.joinpath(*parts[parts.index(root.name)+1:])
    target=(root/path).resolve()
    if not target.is_relative_to(root.resolve()):raise ValueError('Audio path outside workspace')
    return target


def load_config(root, path):
    raw = json.loads(path.read_text()); c = ExperimentConfig(**raw)
    protocol = json.loads(safe_path(root, c.protocol_path).read_text())
    if protocol['config_hashes'].get(c.experiment_id) != object_hash(raw) or protocol['calibration_enabled'] or protocol['evaluation_enabled']:
        raise ValueError('Config hash mismatch or reserved access')
    return c


def dataset(root, c):
    from .datasets import load_public_manifest
    from .provenance import audit_partitions
    for path, h in ((c.source_manifest, c.manifest_sha256), (c.development_manifest, c.development_sha256)):
        if file_hash(safe_path(root, path)) != h: raise ValueError('Source manifest changed')
    train = [r for r in load_public_manifest(root / c.source_manifest) if r.split == 'train']
    dev = [r for r in load_public_manifest(root / c.development_manifest) if r.split == 'development']
    if c.source_manifest == MEDIUM:
        selection_dir = (root / MEDIUM).parent
        b_ids = {r['sample_id'] for r in read_csv(selection_dir / 'selection_rows.csv') if r['stratum'] == 'B_REAL_ONLY'}
    else:
        fake_speakers = {r.speaker_id for r in dev if r.label}
        b_ids = {r.sample_id for r in dev if r.speaker_id not in fake_speakers}
    if c.tiny_per_attack:
        def small(rows):
            chosen = []
            for attack in [None] + [f'A{i:02}' for i in range(1, 7)]:
                pool = sorted([r for r in rows if (r.label == 0 if attack is None else r.label == 1 and r.attack_id == attack)], key=lambda r: r.sample_id)
                n = c.tiny_per_attack * (6 if attack is None else 1)
                if len(pool) < n: raise ValueError('Tiny coverage missing')
                chosen += pool[:n]
            return chosen
        train = small(train)
        primary = small([r for r in dev if r.sample_id not in b_ids])
        dev = primary + sorted([r for r in dev if r.sample_id in b_ids], key=lambda r:r.sample_id)[:2]
    if not train or not dev: raise ValueError('Missing train/dev')
    audit = audit_partitions(train + dev, require_speaker_disjoint=True)
    for r in train + dev:
        group_key(r.label, r.attack_id, 'clean')
        if file_hash(source_path(root, r.path)) != r.checksum: raise ValueError('Audio changed: ' + r.sample_id)
    for rows in (train, [r for r in dev if r.sample_id not in b_ids]):
        if set(r.label for r in rows) != {0, 1}: raise ValueError('Both classes required')
    return sorted(train, key=lambda r:r.sample_id), sorted(dev, key=lambda r:r.sample_id), b_ids, audit


def epoch_order(train, c, epoch):
    rng = np.random.default_rng(np.random.SeedSequence([c.training_seed + 100000, epoch]))
    n = int(np.ceil(len(train) / c.batch_size)) * c.batch_size
    if c.sampling == 'natural': return rng.permutation(len(train)).tolist()
    keys = [str(r.label) if c.sampling == 'class_balanced' else group_key(r.label, r.attack_id, 'source') for r in train]
    groups = {k: np.array([i for i, v in enumerate(keys) if v == k]) for k in sorted(set(keys))}
    if c.sampling == 'group_balanced' and len(groups) != 7: raise ValueError('DRO requires REAL and A01--A06')
    result = []
    for _ in range(0, n, c.batch_size):
        # Each effective update contains every group. Repeats remain source reuse.
        labels = list(groups); rng.shuffle(labels)
        labels = (labels * int(np.ceil(c.batch_size / len(labels))))[:c.batch_size]
        rng.shuffle(labels)
        result.extend(int(rng.choice(groups[k])) for k in labels)
    return result


class Inputs:
    def __init__(self, root, c):
        self.root, self.c, self.memo = root, c, OrderedDict()
        self.wave_source = None
        self.wave_views = None
        self.timings = Counter()
        self.index = {}; self.arrays = None
        if c.trainable_backbone_policy == 'HEAD_ONLY':
            from .phase7_data import load_cache
            cfg = json.loads((root/'configs/jaes_v2/phase7/cache_medium_v1.json').read_text())
            cache, meta, _ = load_cache(root, cfg)
            stored = np.load(cache/'embeddings.npz', allow_pickle=False)
            self.arrays = {k: stored[k] for k in stored.files}; stored.close()
            self.index = {(r['sample_id'], r['condition']): r for r in read_csv(cache/'view_manifest.csv')}
            self.cache_sha256 = file_hash(cache/'cache_metadata.json')
        else: self.cache_sha256 = None

    def get(self, r, condition):
        if self.arrays is not None:
            row = self.index[(r.sample_id, condition)]
            return self.arrays[row['array_key']][int(row['array_index'])]
        key = (r.sample_id, condition)
        if key in self.memo:
            self.memo.move_to_end(key); return self.memo[key]
        from .transforms import load_audio
        from .controls import prepare_headroom
        from .phase7_data import waveform_views
        from .preprocessing import Preprocessor
        t = perf_counter()
        if condition == 'clean':
            a = load_audio(source_path(self.root,r.path), sample_rate=22050, max_seconds=3)
            self.timings['decode_seconds'] += perf_counter()-t; t = perf_counter()
            a, _ = prepare_headroom(a, .25)
        else:
            from dataclasses import replace
            if self.wave_source!=r.sample_id:
                local=replace(r,path=str(source_path(self.root,r.path)))
                self.wave_views, _ = waveform_views(self.root, local, shutil.which('ffmpeg'))
                self.wave_source=r.sample_id
            a = self.wave_views[condition]
        self.timings['transform_seconds'] += perf_counter()-t; t = perf_counter()
        x = Preprocessor('P0').extract(a.samples)
        self.timings['mel_seconds'] += perf_counter()-t
        if self.c.resource_limits['tensor_lru_items']:
            self.memo[key] = x
            while len(self.memo) > self.c.resource_limits['tensor_lru_items']: self.memo.popitem(last=False)
        return x


def model_factory(tf, c):
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(c.training_seed)
    from src.model import build_hybrid_model, unfreeze_backbone_for_finetuning
    full, backbone = build_hybrid_model(weights='imagenet')
    if c.trainable_backbone_policy == 'WARMUP_PARTIAL': unfreeze_backbone_for_finetuning(backbone)
    extractor = tf.keras.Model(full.inputs, full.get_layer('time_distributed_mobilenetv3small').output)
    # The exported standard architecture retains its Dropout layer. Training
    # bypasses that stochastic layer using a stateless, explicitly keyed mask.
    def forward(x, training=False, counter=0, stream=0):
        z = x if c.trainable_backbone_policy == 'HEAD_ONLY' else extractor(x, training=training)
        z = full.get_layer('temporal_lstm')(z, training=training)
        z = full.get_layer('classifier_dense')(z)
        if training:
            z = tf.nn.experimental.stateless_dropout(z, rate=full.get_layer('classifier_dropout').rate,
                seed=tf.stack([tf.cast(c.training_seed + 400000 + stream, tf.int32), tf.cast(counter, tf.int32)]))
        return full.get_layer('probability_fake')(z)
    return full, backbone, forward


def variable_hash(variables):
    import hashlib
    h = hashlib.sha256()
    for v in variables:
        h.update(str(v.shape).encode()); h.update(v.numpy().tobytes())
    return h.hexdigest()


def verify_tree(directory, seal='seal.json'):
    m = json.loads((directory/seal).read_text())
    for name, h in m['output_hashes'].items():
        if file_hash(safe_path(directory, name)) != h: raise ValueError('Artifact changed: ' + name)
    return m


def seal_tree(directory):
    write_json(directory/'seal.json', {'output_hashes': {p.relative_to(directory).as_posix(): file_hash(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'seal.json'}})


def save_checkpoint(tf, full, optimizer, directory, state):
    if directory.exists(): raise FileExistsError(directory)
    import tempfile
    directory.parent.mkdir(parents=True,exist_ok=True)
    temporary=Path(tempfile.mkdtemp(prefix=directory.name+'.pending.',dir=directory.parent))
    checkpoint = tf.train.Checkpoint(model=full, optimizer=optimizer)
    checkpoint.write(str(temporary/'state'))
    write_json(temporary/'controller.json', state)
    seal_tree(temporary); os.replace(temporary, directory)


def restore_checkpoint(tf, full, optimizer, directory, contract):
    verify_tree(directory)
    state = json.loads((directory/'controller.json').read_text())
    if state['contract'] != contract: raise ValueError('Resume contract mismatch (config/data/code/environment)')
    tf.train.Checkpoint(model=full, optimizer=optimizer).read(str(directory/'state')).assert_consumed()
    if int(optimizer.iterations.numpy()) != state['step']: raise ValueError('Optimizer cursor mismatch')
    return state


def evaluate(tf, forward, inputs, records, b_ids, conditions, guard=None):
    from .phase7_training import score_metrics
    infer = tf.function(lambda x: forward(x, training=False), reduce_retracing=True)
    rows = []; start = perf_counter(); first_parity = None
    model_seconds=0.
    # Source-first traversal reuses one deterministic waveform bank. It avoids
    # regenerating every codec chain for each condition and bounds cache memory.
    for r in records:
        if guard:guard()
        for condition in conditions:
            x = inputs.get(r, condition)[None]
            t=perf_counter()
            p = float(infer(x).numpy().reshape(-1)[0])
            if first_parity is None:
                first_parity = abs(p-float(forward(x, training=False).numpy().reshape(-1)[0]))
                if first_parity > 1e-6: raise ValueError('Compiled/eager inference parity')
            model_seconds+=perf_counter()-t
            rows.append(dict(sample_id=r.sample_id,speaker_id=r.speaker_id,attack_id=r.attack_id,label=r.label,
                condition_id=condition,stratum='B_REAL_ONLY' if r.sample_id in b_ids else 'A_PAIRED',raw_score=p,predicted_label=int(p>=.5),threshold=.5))
    metric_start=perf_counter();metrics = []
    for condition in conditions:
        for stratum in ('A_PAIRED','B_REAL_ONLY'):
            group = [r for r in rows if r['condition_id']==condition and r['stratum']==stratum]
            if group: metrics.append(dict(condition=condition,stratum=stratum,**score_metrics(group)))
    return rows, metrics, {'validation_seconds':perf_counter()-start,'model_forward_seconds':model_seconds,
        'metric_seconds':perf_counter()-metric_start,'compiled_eager_residual':first_parity}


def selection_value(metrics, c):
    clean = next(m for m in metrics if m['condition']=='clean' and m['stratum']=='A_PAIRED')
    if c.checkpoint_policy == 'CLEAN_BCE': return clean['nll'], 'ELIGIBLE'
    if clean['REAL_false_alarm'] > c.clean_REAL_FA_cap or clean['FAKE_miss'] > c.clean_FAKE_miss_cap:
        return None, 'INFEASIBLE_UNDER_REGISTERED_CONSTRAINT'
    stress = [m for m in metrics if m['condition'] in ('noise5','A_then_B','B_then_A') and m['stratum']=='A_PAIRED']
    if len(stress)!=3: raise ValueError('Guarded selection requires measured stress metrics')
    return float(np.mean([.5*(m['REAL_false_alarm']+m['FAKE_miss']) for m in stress])), 'ELIGIBLE'


def train(root, c, *, resume=False, stop_after=None):
    from .c2_student import peak_process_mib
    tf = initialize(c.runtime); start = perf_counter()
    train_rows, dev, b_ids, data_audit = dataset(root, c)
    inputs = Inputs(root, c); full, backbone, forward = model_factory(tf, c)
    variables = list(full.trainable_weights)
    backbone_ids = {id(v) for v in backbone.trainable_weights}
    frozen = [v for v in full.weights if id(v) not in {id(x) for x in variables}]
    frozen_hash = variable_hash(frozen)
    backbone_before = variable_hash(backbone.trainable_weights)
    bn = [l for l in backbone.layers if isinstance(l,tf.keras.layers.BatchNormalization)]
    if not all(not l.trainable for l in bn): raise ValueError('BN must be frozen')
    optimizer = tf.keras.optimizers.Adam(c.learning_rate); optimizer.build(variables)
    out = root/BASE/'training'/c.experiment_id
    contract = dict(config_hash=object_hash(asdict(c)),code=code_contract(root),manifest=c.manifest_sha256,
        development=c.development_sha256,cache=inputs.cache_sha256,tf=tf.__version__,python=platform.python_version(),runtime=c.runtime,
        actual_sources=object_hash([r.to_dict() for r in train_rows+dev]))
    stages = c.warmup_epochs+c.finetune_epochs
    keys = sorted({group_key(r.label,r.attack_id,condition) for r in train_rows for condition in BANKS[c.augmentation_policy]})
    state = dict(contract=contract,step=0,epoch=0,cursor=0,micro_counter=0,best=None,best_epoch=None,bad_epochs=0,
        history=[],update_chain=object_hash([]),dro_weights=[1/len(keys)]*len(keys),group_keys=keys,dropout='STATELESS_COUNTER',
        rng_policy='SeedSequence(seed,epoch); view SeedSequence(augmentation_seed,epoch,update); no mutable RNG consumed',
        gradient_accumulation='zero at checkpoint boundary',pending_validation=False,finished=False,
        initialization_hash=variable_hash(full.weights),frozen_hash=frozen_hash)
    if resume:
        state = restore_checkpoint(tf,full,optimizer,out/'checkpoints'/json.loads((out/'latest.json').read_text())['checkpoint'],contract)
        if (out/'seal.json').exists(): raise ValueError('Sealed run cannot resume; verify instead')
        if state['frozen_hash']!=frozen_hash: raise ValueError('Frozen initialization changed')
        chain=object_hash([])
        for step in range(1,state['step']+1):
            row=json.loads((out/'updates'/f'{step:08}.json').read_text());chain=object_hash([chain,row])
        if chain!=state['update_chain']:raise ValueError('Committed source/view journal changed')
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(out/'resolved_config.json',asdict(c));write_json(out/'contract.json',contract)
        write_json(out/'dataset_audit.json',dict(partitions=data_audit,unique_REAL_sources=sum(r.label==0 for r in train_rows),unique_FAKE_sources=sum(r.label==1 for r in train_rows),sampling=c.sampling,source_ids=[r.sample_id for r in train_rows],development_ids=[r.sample_id for r in dev],calibration_accessed=False,evaluation_accessed=False))
        write_json(out/'environment.json',dict(python=sys.version,tensorflow=tf.__version__,GPU=[str(x) for x in tf.config.list_physical_devices('GPU')],TEMP=os.getenv('TEMP'),mixed_precision='float32'))
        write_json(out/'trainable_layers.json',dict(variables=[dict(name=v.name,shape=v.shape.as_list()) for v in variables],BN_frozen=[l.name for l in bn],trainable_parameters=sum(int(np.prod(v.shape)) for v in variables),initialization_hash=state['initialization_hash']))
    timing = Counter(startup_seconds=perf_counter()-start); invocation_updates=0; last_gradients={}

    def guard():
        if shutil.disk_usage(root).free < c.resource_limits['disk_reserve_gib']*1024**3+64*1024**2: raise RuntimeError('Disk reserve reached')
        peak=peak_process_mib()
        if peak and peak>c.resource_limits['max_process_mib']: raise RuntimeError('RAM budget reached')
        if perf_counter()-start>c.resource_limits['seconds_per_invocation']: raise TimeoutError('Invocation budget reached; resume last committed boundary')

    def commit():
        tag=f"e{state['epoch']:03}_u{state['step']:08}_c{state['cursor']:06}_{int(state['pending_validation'])}_{int(state['finished'])}"
        t=perf_counter()
        target=out/'checkpoints'/tag
        if target.exists():
            verify_tree(target)
            if json.loads((target/'controller.json').read_text())!=json.loads(json.dumps(state)): raise ValueError('Orphan checkpoint state differs')
        else: save_checkpoint(tf,full,optimizer,target,state)
        write_json(out/'latest.json',dict(checkpoint=tag,controller_sha256=file_hash(out/'checkpoints'/tag/'controller.json')))
        timing['checkpoint_seconds']+=perf_counter()-t

    @tf.function(reduce_retracing=True)
    def gradient(x,y,coeff,counter,clean,paired):
        with tf.GradientTape() as tape:
            p=forward(x,True,counter,0)
            value=weighted_sum(y,p,coeff)
            if paired:
                q=forward(clean,True,counter,1)
                value=.5*(value+weighted_sum(y,q,coeff))
                if c.training_objective=='CONSISTENCY':
                    value+=c.objective_parameters['consistency_weight']*tf.reduce_sum(tf.square(tf.reshape(p,[-1])-tf.reshape(q,[-1]))*coeff)
        gradients=tape.gradient(value,variables)
        for g in gradients:
            if g is None: raise ValueError('Detached variable')
            tf.debugging.assert_all_finite(g,'Nonfinite gradient')
        return value,gradients,bce(y,p)

    try:
        while state['epoch']<stages and not state['finished']:
            epoch=state['epoch']; partial=epoch>=c.warmup_epochs
            lrbase=c.finetune_learning_rate if partial else c.learning_rate
            stage_epoch=epoch-c.warmup_epochs if partial else epoch
            stage_total=c.finetune_epochs if partial else c.warmup_epochs
            lr=lrbase*(1-.5*stage_epoch/stage_total);optimizer.learning_rate.assign(lr)
            order=epoch_order(train_rows,c,epoch)
            while state['cursor']<len(order):
                guard()
                if state['step']>=c.maximum_updates: raise RuntimeError('Maximum updates reached before declared lifecycle completion')
                chosen=[train_rows[i] for i in order[state['cursor']:state['cursor']+c.batch_size]]
                bank=BANKS[c.augmentation_policy]; all_views=c.training_objective in ('EQUAL_VIEW','GROUP_DRO')
                rng=np.random.default_rng(np.random.SeedSequence([c.training_seed+200000,epoch,state['cursor']]))
                views=[(r,condition) for r in chosen for condition in (bank if all_views else [str(rng.choice(bank))])]
                group_ids=[keys.index(group_key(r.label,r.attack_id,condition)) for r,condition in views]
                counts=Counter(group_ids)
                if c.training_objective=='GROUP_DRO':
                    if len(counts)!=len(keys): raise ValueError('Missing DRO group in effective update')
                    coeff=np.array([state['dro_weights'][g]/counts[g] for g in group_ids],dtype='float32')
                elif c.training_objective=='WEIGHTED_BCE':
                    weights=c.objective_parameters['class_weights'];coeff=np.array([weights[r.label]/len(views) for r,_ in views],dtype='float32')
                else: coeff=np.full(len(views),1/len(views),dtype='float32')
                acc=[tf.zeros_like(v) for v in variables]; loss=0.;group_loss=np.zeros(len(keys));t=perf_counter()
                for begin in range(0,len(views),c.microbatch_size):
                    chunk=views[begin:begin+c.microbatch_size]
                    x=np.stack([inputs.get(r,condition) for r,condition in chunk]);y=np.array([r.label for r,_ in chunk],dtype='float32')
                    paired=c.training_objective=='CONSISTENCY'
                    clean=np.stack([inputs.get(r,'clean') for r,_ in chunk]) if paired else x
                    val,gs,ls=gradient(x,y,coeff[begin:begin+len(chunk)],tf.constant(state['micro_counter'],tf.int32),clean,paired)
                    acc=[a+g for a,g in zip(acc,gs)];loss+=float(val)
                    for j,ell in enumerate(ls.numpy()):group_loss[group_ids[begin+j]]+=float(ell)
                    state['micro_counter']+=1
                active=[(g,v) for g,v in zip(acc,variables) if partial or id(v) not in backbone_ids]
                optimizer.apply_gradients(active);state['step']+=1; invocation_updates+=1
                timing['update_seconds']+=perf_counter()-t
                last_gradients={v.name:float(tf.linalg.norm(g)) for g,v in active}
                if c.training_objective=='GROUP_DRO':
                    risks=np.array([group_loss[g]/counts[g] for g in range(len(keys))])
                    state['dro_weights']=dro_update(state['dro_weights'],risks,c.objective_parameters['dro_rate'],c.objective_parameters['dro_floor']).tolist()
                entry=dict(step=state['step'],epoch=epoch,learning_rate=lr,loss=loss,
                    sample_ids=[r.sample_id for r in chosen],views=[(r.sample_id,condition) for r,condition in views],
                    micro_counter=state['micro_counter'],supervised_views=len(views)*(2 if paired else 1),backward_calls=int(np.ceil(len(views)/c.microbatch_size)))
                journal=out/'updates'/f"{state['step']:08}.json";journal.parent.mkdir(exist_ok=True)
                if journal.exists() and json.loads(journal.read_text())!=json.loads(json.dumps(entry)):raise ValueError('Replayed update differs from orphan journal')
                write_json(journal,entry);state['update_chain']=object_hash([state['update_chain'],entry])
                state['cursor']+=len(chosen)
                state['pending_validation']=state['cursor']>=len(order)
                if variable_hash(frozen)!=frozen_hash: raise ValueError('Frozen weights/BN changed')
                if state['step'] % c.checkpoint_every_updates==0 or state['pending_validation'] or (stop_after and invocation_updates>=stop_after): commit()
                if stop_after and invocation_updates>=stop_after:
                    write_json(out/'status.json',dict(status='INTERRUPTED_AT_COMMITTED_BOUNDARY',step=state['step']))
                    return state
            # A pending validation is replayed after interruption; no extra optimizer update.
            primary=[r for r in dev if r.sample_id not in b_ids]
            conditions=('clean',) if c.checkpoint_policy=='CLEAN_BCE' else ('clean','noise5','A_then_B','B_then_A')
            _,metrics,measure=evaluate(tf,forward,inputs,primary,b_ids,conditions,guard)
            timing.update(measure);value,eligibility=selection_value(metrics,c)
            improved=value is not None and (state['best'] is None or value<state['best'])
            state['history'].append(dict(epoch=epoch+1,step=state['step'],learning_rate=lr,value=value,eligibility=eligibility,metrics=metrics))
            if improved:
                state['best']=value;state['best_epoch']=epoch+1;state['bad_epochs']=0
                best=out/f'best_epoch_{epoch+1:03}.keras';tmp=best.with_suffix('.temporary.keras');full.save(tmp);os.replace(tmp,best)
                write_json(out/'best.json',dict(path=best.name,sha256=file_hash(best),epoch=epoch+1,selection_value=value,criterion=c.checkpoint_policy))
            else: state['bad_epochs']+=1
            state['epoch']+=1;state['cursor']=0;state['pending_validation']=False
            state['finished']=state['epoch']>=stages or state['bad_epochs']>=c.early_stopping_patience
            commit();write_json(out/'history.json',state['history'])
            print(f"{c.experiment_id} epoch={state['epoch']} step={state['step']} selection={value} {eligibility}",flush=True)
        if state['best'] is None:
            write_json(out/'status.json',dict(status='NO_FEASIBLE_CHECKPOINT',steps=state['step']));return state
        # Evaluation reloads the selected full model; head path extracts its layers.
        current_hash=variable_hash(full.weights)
        best=json.loads((out/'best.json').read_text())
        if file_hash(out/best['path'])!=best['sha256']:raise ValueError('Best export changed')
        loaded=tf.keras.models.load_model(out/best['path'],compile=False)
        if c.trainable_backbone_policy=='HEAD_ONLY':
            def selected(x,training=False):
                z=x
                for name in ('temporal_lstm','classifier_dense','probability_fake'):z=loaded.get_layer(name)(z)
                return z
        else: selected=lambda x,training=False:loaded(x,training=False)
        rows,metrics,measure=evaluate(tf,selected,inputs,dev,b_ids,DEV,guard);timing.update(measure)
        write_csv(out/'scores.csv',rows);write_json(out/'metrics.json',metrics)
        # Per-attack FAKE-only metrics retain null AUC/FAR rather than borrowing REAL rows.
        from .phase7_training import score_metrics
        subgroups=[]
        for condition in DEV:
            for attack in [f'A{i:02}' for i in range(1,7)]:
                group=[r for r in rows if r['condition_id']==condition and r['attack_id']==attack]
                if group:subgroups.append(dict(condition=condition,attack=attack,**score_metrics(group)))
        write_json(out/'attack_metrics.json',subgroups)
        write_json(out/'resource_measurements.json',dict(elapsed_seconds=perf_counter()-start,invocation_updates=invocation_updates,
            timing=dict(timing),input_timing=dict(inputs.timings),peak_process_mib=peak_process_mib(),GPU_memory=tf.config.experimental.get_memory_info('GPU:0') if c.runtime=='gpu' else None,
            retained_checkpoint_bytes=sum(p.stat().st_size for p in (out/'checkpoints').rglob('*') if p.is_file()),last_gradients=last_gradients,
            final_weights_hash=current_hash,optimizer_hash=variable_hash(optimizer.variables()),frozen_unchanged=variable_hash(frozen)==frozen_hash,
            backbone_changed_in_invocation=backbone_before!=variable_hash(backbone.trainable_weights)))
        write_json(out/'status.json',dict(status='PILOT_EVALUATED' if c.tiny_per_attack else 'DEVELOPMENT_EVALUATED',steps=state['step'],best=best,calibration_accessed=False,evaluation_accessed=False))
        seal_tree(out);return state
    except BaseException as exc:
        write_json(out/'failure_'+str(os.getpid())+'.json',dict(type=type(exc).__name__,message=str(exc),last_committed=json.loads((out/'latest.json').read_text()) if (out/'latest.json').exists() else None))
        raise
