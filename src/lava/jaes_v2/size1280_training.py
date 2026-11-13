"""Size-1280 lifecycle using the frozen Phase 10 numerical and artifact functions."""
from collections import Counter
from time import perf_counter

import numpy as np

from scripts.jaes_v2 import size1280_profile as profile
from . import phase10_training as engine, phase8_training as prior
from .evidence import file_hash, object_hash, write_csv, write_json
from .phase10_resources import ResourceStop, snapshot
from .phase10_validation import guard_counts, metrics, score


def update_plan(train, config, epoch, cursor, order):
    rng = np.random.default_rng(np.random.SeedSequence([config.training_seed + 200000, epoch, cursor]))
    return [(train[i], str(rng.choice(prior.BANKS[config.augmentation_policy])))
            for i in order[cursor:cursor + config.batch_size]]


def learning_rate(config, epoch):
    if epoch < config.warmup_epochs:
        return config.learning_rate
    return config.finetune_learning_rate * (1 - .5 * (epoch - config.warmup_epochs) / config.finetune_epochs)


def contract(root, spec, config, method, role, data, release):
    train, dev, b_ids, _ = data
    _, baseline = profile.reference(root, method)
    if baseline['initial_weights'] != spec['initial_weights']:
        raise ValueError('Registered initialization differs from the sealed 640 arm')
    return dict(profile=profile.PROFILE, experiment=spec['experiments'][method],
                run_kind='scientific' if role == 'scientific' else 'qualification',
                release=release, config=config.__dict__, runtime=profile.runtime(),
                train_manifest=spec['train_manifest'], train_manifest_sha256=spec['train_manifest_sha256'],
                actual_sources=object_hash([r.to_dict() for r in train + dev]), b_ids=sorted(b_ids),
                initial_weights=spec['initial_weights'], backend=engine.BACKEND, threshold=.5,
                policy='CLEAN_BCE', tier_b_every=3, calibration_used=False, official_evaluation_used=False,
                augmentation_rng='SeedSequence(training_seed+200000,epoch,cursor)',
                dropout_rng='stateless training_seed and micro counter')


def initialize(root, spec, config, expected):
    profile.weight_cache(spec)
    tf = engine.phase10_tensorflow()
    model, backbone, forward = prior.model_factory(tf, config)
    if prior.variable_hash(model.weights) != expected:
        raise ValueError('Fresh initialization differs from seed 9207 fingerprint')
    if any(layer.trainable for layer in backbone.layers if isinstance(layer, tf.keras.layers.BatchNormalization)):
        raise ValueError('BatchNorm must stay frozen')
    return tf, model, backbone, forward


def restore(tf, model, optimizer, checkpoint, state):
    tf.train.Checkpoint(model=model, optimizer=optimizer).read(str(checkpoint / 'state')).assert_consumed()
    if (int(optimizer.iterations.numpy()) != state['step']
            or prior.variable_hash(model.weights) != state['model_hash']
            or prior.variable_hash(optimizer.variables()) != state['optimizer_hash']):
        raise ValueError('Restored tensors or optimizer counter mismatch')


class Run:
    def __init__(self, root, spec, config, method, role, data, release):
        self.root, self.spec, self.config = root, spec, config
        self.train, self.dev, self.b_ids, self.audit = data
        self.out = profile.directory(root, spec, method, role)
        self.role = role
        self.contract = contract(root, spec, config, method, role, data, release)
        self.started = perf_counter()
        self.tf, self.model, self.backbone, forward = initialize(root, spec, config, self.contract['initial_weights'])
        tf = self.tf
        self.warmup, self.partial, self.variables, self.cnn, _ = engine.kernel(tf, self.backbone, self.model, forward)
        self.optimizer = tf.keras.optimizers.Adam(config.learning_rate)
        self.optimizer.build(self.variables)
        self.frozen = [v for v in self.model.weights if id(v) not in {id(v) for v in self.variables}]
        self.frozen_hash = prior.variable_hash(self.frozen)
        self.cnn_variables = [v for v in self.variables if id(v) in self.cnn]
        self.initial_cnn_hash = prior.variable_hash(self.cnn_variables)
        self.infer = prior.segmentwise_inference(tf, self.model)
        self.inputs = prior.Inputs(root, config)
        self.timing = Counter(startup_and_restore=0, train_seconds=0, validation_seconds=0, checkpoint_seconds=0,
                              export_seconds=0)
        self.samples = []
        self.updates = 0
        self.state = dict(contract=self.contract, step=0, epoch=0, cursor=0, micro=0,
                          journal=[], history=[], best=None, bad_epochs=0, warning=None, frozen_hash=self.frozen_hash)

    def guard(self, write_bytes=0):
        if perf_counter() - self.started > self.config.resource_limits['seconds_per_invocation']:
            raise ResourceStop('INVOCATION_TIME_LIMIT')
        return profile.guard_disk(self.root, write_bytes)

    def commit(self):
        state, out = self.state, self.out
        t = perf_counter()
        state.update(model_hash=prior.variable_hash(self.model.weights),
                     optimizer_hash=prior.variable_hash(self.optimizer.variables()))
        tag = f"u{state['step']:06d}_e{state['epoch']:03d}_c{state['cursor']:03d}"
        target = out / 'checkpoints' / tag
        if target.exists():
            if engine.strict_controller(target, self.contract) != state:
                raise ValueError('Orphan checkpoint differs from deterministic replay')
        else:
            budget = self.spec['resource_plan']['checkpoint_mib'] * 1024 ** 2
            self.guard(budget)
            prior.save_checkpoint(self.tf, self.model, self.optimizer, target, state)
            if sum(p.stat().st_size for p in target.iterdir() if p.is_file()) > budget:
                raise ResourceStop('CHECKPOINT_EXCEEDED_STORAGE_PLAN')
        write_json(out / 'latest.json', dict(checkpoint=tag, sha256=file_hash(target / 'controller.json')))
        self.timing['checkpoint_seconds'] += perf_counter() - t

    def publish(self, status):
        state, out = self.state, self.out
        seconds = self.timing['train_seconds']
        write_json(out / 'status.json', dict(status=status, step=state['step'], epoch=state['epoch'],
            last_update=state['journal'][-1] if state['journal'] else None,
            validation=state['history'][-1] if state['history'] else None, best=state['best'],
            early_stopping_counter=state['bad_epochs'], warning=state['warning'],
            latest=engine.read(out / 'latest.json') if (out / 'latest.json').is_file() else None,
            resources=self.samples[-1] if self.samples else None, gpu_vram=None,
            updates_per_second=self.updates / seconds if seconds else None,
            remaining_train_seconds=(self.config.maximum_updates - state['step']) * seconds / self.updates
            if self.updates else None))

    def update(self, views, epoch):
        """Apply the released kernel, accumulation order, Adam slots and stage mask."""
        tf, config, state = self.tf, self.config, self.state
        partial = epoch >= config.warmup_epochs
        lr = learning_rate(config, epoch)
        self.optimizer.learning_rate.assign(lr)
        active = [i for i, v in enumerate(self.variables) if partial or id(v) not in self.cnn]
        accum = [tf.zeros_like(v) for v in self.variables]
        loss = 0.
        gradient = self.partial if partial else self.warmup
        for record, condition in views:
            value, grads = gradient(self.inputs.get(record, condition)[None], tf.constant([record.label]),
                                    tf.constant([1 / len(views)], tf.float32), tf.constant(state['micro']))
            tf.debugging.assert_all_finite(value, 'Nonfinite loss')
            for grad in grads:
                tf.debugging.assert_all_finite(grad, 'Invalid micro-gradient')
            accum = [a + g for a, g in zip(accum, grads, strict=True)]
            loss += float(value.numpy())
            state['micro'] += 1
        norm = float(tf.linalg.global_norm([accum[i] for i in active]).numpy())
        if not np.isfinite(norm) or not 0 < norm <= 1e5 or loss > 20:
            raise FloatingPointError(f'Invalid update norm={norm} loss={loss}')
        active_weights = [self.variables[i] for i in active]
        before = prior.variable_hash(active_weights)
        self.optimizer.apply_gradients((accum[i], self.variables[i]) for i in active)
        for variable in active_weights:
            tf.debugging.assert_all_finite(variable, 'Invalid updated weight')
        if before == prior.variable_hash(active_weights):
            raise FloatingPointError('No active weight changed')
        state['step'] += 1
        state['cursor'] += len(views)
        state['journal'].append(dict(step=state['step'], epoch=epoch, lr=lr, loss=loss, norm=norm,
                                     sources=[[r.sample_id, view] for r, view in views], micro=state['micro']))

    def validate(self, epoch, final):
        state = self.state
        is_b = epoch % 3 == 0 or final
        tier = 'B' if is_b else 'A'
        rows, measured = score(self.dev, self.b_ids, tier, self.inputs, {'fixed': self.infer}, self.guard)
        self.timing['validation_seconds'] += measured['elapsed_seconds']
        rows = rows['fixed']
        clean = metrics(rows)['A_PAIRED/clean']
        collapsed = clean['fp'] == clean['n_REAL'] or clean['fn'] == clean['n_FAKE']
        previous = state['history'][-1]['collapse_epochs'] if state['history'] else 0
        collapse_epochs = previous + 1 if collapsed else 0
        if collapse_epochs >= 2 and epoch >= 3:
            state['warning'] = 'PERSISTENT_PREDICTION_COLLAPSE'
        state['history'].append(dict(epoch=epoch, step=state['step'], tier=tier, clean=clean,
            collapse_epochs=collapse_epochs, guard_counts=guard_counts(rows) if is_b else None, guard=None))
        rank = [clean['nll'], clean['nll'], epoch]
        if state['best'] is None or rank < state['best']['rank']:
            state['best'] = dict(rank=rank, tag=f"u{state['step']:06d}_e{state['epoch']:03d}_c{state['cursor']:03d}")
            state['bad_epochs'] = 0
        else:
            state['bad_epochs'] += 1

    def finish(self):
        state, out = self.state, self.out
        if prior.variable_hash(self.frozen) != self.frozen_hash:
            raise FloatingPointError('Frozen model state changed')
        selected = out / 'checkpoints' / state['best']['tag']
        selected_state = engine.strict_controller(selected, self.contract)
        restore(self.tf, self.model, self.optimizer, selected, selected_state)
        rows, measured = score(self.dev, self.b_ids, 'C', self.inputs, {'fixed': self.infer}, self.guard)
        self.timing['validation_seconds'] += measured['elapsed_seconds']
        rows = rows['fixed']
        t = perf_counter()
        self.guard(32 * 1024 ** 2)
        self.model.save(out / 'selected.keras')
        self.timing['export_seconds'] += perf_counter() - t
        write_csv(out / 'scores.csv', rows)
        write_json(out / 'result.json', dict(state=state, final_weights=state['model_hash'],
                   selected_weights=selected_state['model_hash'], scores_hash=object_hash(rows), metrics=metrics(rows)))
        self.publish('SCIENTIFIC_COMPLETE' if self.role == 'scientific' else 'QUALIFICATION_COMPLETE')
        prior.seal_tree(out)

    def execute(self, *, resume=False, stop_at=None):
        state, out, c, tf = self.state, self.out, self.config, self.tf
        if resume:
            checkpoint, state = engine.committed_controller(out, self.contract)
            if state['frozen_hash'] != self.frozen_hash:
                raise ValueError('Frozen state differs from the registered initialization')
            restore(tf, self.model, self.optimizer, checkpoint, state)
            self.state = state
            if stop_at is not None and stop_at <= state['step']:
                raise ValueError('Stop step must advance the checkpoint')
            if state['warning'] == 'UNEXPECTED_THROUGHPUT_STALL':
                state['warning'] = None
            write_json(out / 'resume_events' / f"u{state['step']:06d}.json",
                       dict(step=state['step'], model_hash=state['model_hash'], optimizer_hash=state['optimizer_hash'],
                            controller_sha256=file_hash(checkpoint / 'controller.json')))
        else:
            out.mkdir(parents=True, exist_ok=False)
            write_json(out / 'contract.json', self.contract)
            write_json(out / 'data_audit.json', self.audit)
        self.timing['startup_and_restore'] = perf_counter() - self.started
        try:
            if not resume:
                self.commit()
            self.publish('RUNNING')
            order = None
            while state['step'] < c.maximum_updates:
                self.samples.append(self.guard())
                epoch, cursor = state['epoch'], state['cursor']
                if order is None:
                    order = prior.epoch_order(self.train, c, epoch)
                views = update_plan(self.train, c, epoch, cursor, order)
                started = perf_counter()
                self.update(views, epoch)
                elapsed = perf_counter() - started
                self.timing['train_seconds'] += elapsed
                self.timing['warmup_seconds' if epoch < c.warmup_epochs else 'partial_seconds'] += elapsed
                self.updates += 1
                if elapsed > 300:
                    state['warning'] = 'UNEXPECTED_THROUGHPUT_STALL'
                boundary = state['cursor'] == len(order)
                final = state['step'] == c.maximum_updates
                if boundary:
                    if epoch + 1 == c.warmup_epochs:
                        state['warmup_cnn_hash'] = prior.variable_hash(self.cnn_variables)
                        if state['warmup_cnn_hash'] != self.initial_cnn_hash:
                            raise FloatingPointError('Warm-up changed the CNN')
                    state.update(epoch=epoch + 1, cursor=0)
                    order = None
                if epoch == c.warmup_epochs and cursor == 0:
                    state['first_partial_cnn_hash'] = prior.variable_hash(self.cnn_variables)
                    if state['first_partial_cnn_hash'] == self.initial_cnn_hash:
                        raise FloatingPointError('First partial update did not change the CNN')
                if boundary or final:
                    self.validate(epoch + 1, final)
                pause = state['step'] == stop_at or state['warning'] == 'UNEXPECTED_THROUGHPUT_STALL'
                if final or boundary or pause or state['step'] % c.checkpoint_every_updates == 0:
                    self.commit()
                if pause and not final:
                    self.publish('PAUSE_AT_CHECKPOINT')
                    return
                self.publish('WARNING' if state['warning'] else 'RUNNING')
            self.finish()
        except (ResourceStop, tf.errors.ResourceExhaustedError, MemoryError) as exc:
            state['warning'] = str(exc) if isinstance(exc, ResourceStop) else 'TENSOR_ALLOCATION_FAILURE'
            self.publish('STOP_FOR_RESOURCE_SAFETY')
            raise
        except (FloatingPointError, tf.errors.InvalidArgumentError) as exc:
            state['warning'] = str(exc)
            self.publish('STOP_FOR_NUMERICAL_FAILURE')
            raise
        except KeyboardInterrupt:
            if (out / 'latest.json').is_file():
                self.state = engine.committed_controller(out, self.contract)[1]
                self.publish('PAUSE_AT_CHECKPOINT')
            raise
        finally:
            # Wall time and resource samples are invocation-specific and outside the seal.
            evidence = out.parent / 'resources' / f'{self.role}_{self.started.hex().replace(".", "_")}.json'
            write_json(evidence, dict(timing=self.timing, input_timing=self.inputs.timings, samples=self.samples,
                updates=self.updates, final_step=state['step'], final_resources=snapshot(self.root)))
