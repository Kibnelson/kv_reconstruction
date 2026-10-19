"""Portable CPU-reference registry of the concrete KV representations tested.

Learned state MUST be supplied by calibration on the target model. This module
never loads bundled SmolLM dictionaries. GPU-native packets use separate APIs
and names: these inherited wire formats are CPU-reference implementations.
"""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import numpy as np


def _specs(grid=False):
    specs = {}
    def add(name, kind, **options):
        specs[name] = dict(kind=kind, **options)
    def wrap(name):
        add(name + '_deflate', 'wrap', parent=name)
    add('raw', 'raw', shuffle=False)
    add('raw_paged', 'simple', method='raw')
    wrap('raw')
    add('byte_shuffle', 'raw', shuffle=True)
    wrap('byte_shuffle')
    for method in ('exponent', 'zero_coset', 'lattice2_coset', 'lattice4_coset', 'lattice2_sign_coset'):
        add(method, 'simple', method=method)
    wrap('exponent')
    for strategy in ('packed', 'packed_deflate', 'deflate', 'best'):
        add('field_' + strategy, 'field', strategy=strategy)
    wrap('field_best')
    add('exponent_huffman', 'huffman')
    wrap('exponent_huffman')
    for strategy in ('deflate', 'split_deflate', 'huffman'):
        add('exponent_huffman_lower_' + strategy, 'lower_control', strategy=strategy)
    for predictor in ('previous_token', 'previous_channel', 'zero', 'midpoint', 'previous_token_scaled'):
        add('temporal_' + predictor, 'temporal', predictor=predictor)
    for model in ('lower16', 'lower16_conditional'):
        add(model, 'lower_learned', model=model, force=False)
        add(model + '_forced', 'lower_learned', model=model, force=True)
    for size in (16, 64, 256):
        add('learned' + str(size), 'dictionary', model='learned' + str(size))
    add('fixed64', 'dictionary', model='fixed64')
    wrap('learned64')
    add('hybrid_group', 'hybrid', model='learned64')
    add('hybrid_tile', 'hybrid_tile', model='learned64')
    add('e8_float', 'original_e8', method='e8_float')
    add('e8_symbol', 'original_e8', method='e8_symbol')
    add('direct_entropy', 'direct')
    for assignment in ('radius', 'l1'):
        for correction in ('coset', 'best_entropy', 'best'):
            label = {'coset': 'coset', 'best_entropy': 'entropy', 'best': 'best'}[correction]
            add(f'learned_{label}_{assignment}', 'residual', model='learned64', assignment=assignment, correction=correction)
        wrap('learned_entropy_' + assignment)
    for family in ('e8', 'scalar'):
        for correction in ('coset', 'best'):
            label = 'entropy' if correction == 'best' else 'coset'
            name = f'learned_{family}_{label}'
            add(name, 'residual_e8', model='learned64', family=family, correction=correction, step_log2=10)
            wrap(name)
    for dimension in ((8, 32, 64) if grid else (64,)):
        for transform in ('identity', 'dense', 'hadamard'):
            for precision in ('f32', 'bf16'):
                name = f'binary_{transform}_d{dimension}_{precision}'
                add(name, 'binary', model='learned64', transform=transform, dimension=dimension, amplitude_precision=precision)
                wrap(name)
    if grid:
        for assignment in ('radius', 'l1'):
            for correction in ('raw', 'rice', 'zigzag_deflate', 'centered_deflate', 'centered_packed', 'zigzag_packed'):
                add(f'learned_{correction}_{assignment}', 'residual', model='learned64', assignment=assignment, correction=correction)
    return specs


SMOKE = ('raw', 'exponent_huffman_lower_deflate', 'learned_entropy_l1',
         'binary_identity_d64_bf16', 'binary_hadamard_d64_bf16')
MAIN = ('raw', 'raw_deflate', 'byte_shuffle_deflate', 'exponent', 'exponent_huffman',
        'exponent_huffman_lower_deflate', 'field_best_deflate', 'fixed64',
        'learned64', 'learned64_deflate', 'hybrid_group', 'e8_symbol',
        'learned_coset_radius', 'learned_entropy_radius', 'learned_entropy_l1',
        'learned_e8_entropy', 'learned_e8_entropy_deflate',
        'binary_identity_d64_bf16', 'binary_dense_d64_bf16', 'binary_hadamard_d64_bf16')


def candidate_specs(preset='main', grid=False):
    """Named concrete algorithms; grid adds previously explored parameter arms."""
    specs = _specs(grid)
    if preset not in ('smoke', 'main', 'all'):
        raise ValueError('preset must be smoke, main, or all')
    names = SMOKE if preset == 'smoke' else MAIN if preset == 'main' else tuple(specs)
    return {name: specs[name].copy() for name in names}


def required_artifacts(preset='main', grid=False, names=None):
    specs = candidate_specs(preset, grid)
    if names is not None:
        all_specs = _specs(grid)
        specs = {name: all_specs[name] for name in names}
    needed = set()
    def visit(spec):
        if 'model' in spec:
            needed.add(spec['model'])
        if spec['kind'] == 'wrap':
            visit(_specs(grid)[spec['parent']])
    for spec in specs.values():
        visit(spec)
    return dict(magnitude_sizes=sorted(int(x[7:]) for x in needed if x.startswith('learned')),
        fixed_sizes=sorted(int(x[5:]) for x in needed if x.startswith('fixed')),
        lower_sizes=[16] if 'lower16' in needed else [],
        conditional_lower_sizes=[16] if 'lower16_conditional' in needed else [])


@dataclass
class RegistryRecord:
    data: bytes
    stats: dict


@dataclass
class CodecAdapter:
    name: str
    _encode: object
    _decode: object
    state_bytes: int = 0
    state_info: dict = field(default_factory=dict)
    traits: dict = field(default_factory=dict)
    parent: object = None

    @property
    def dependency(self):
        return self.parent.name if self.parent is not None else None

    def _record(self, value):
        stats = dict(value.stats)
        if 'component_bytes' in stats and sum(stats['component_bytes'].values()) != len(value.data):
            raise AssertionError(self.name + ': incomplete byte accounting')
        counts = dict(stats.get('mechanism_counts', stats.get('branch_diagnostics', {})))
        counts['encoded_records'] = 1
        if 'fallback_counts' in stats:
            counts['reported_fallback_groups'] = int(sum(stats['fallback_counts'].values()))
        modes = stats.get('record_modes', {})
        counts['huffman_fallback_records'] = int(modes.get('original_huffman_fallback', 0))
        raw_records = sum(int(modes.get(mode, 0)) for mode in ('raw', 'raw_smaller', 'raw_nonfinite_or_empty'))
        if self.traits['family'] == 'field':
            raw_records = int(stats.get('mode_id') == 0)
        if self.traits['family'] == 'huffman':
            raw_records = int(stats.get('mode') == 'raw_words')
        counts['raw_fallback_records'] = raw_records
        counts['raw_control_records'] = int(self.traits['family'] == 'raw' or
            (self.traits['family'] == 'simple' and self.traits['options']['method'] == 'raw'))
        if self.traits['family'] in ('simple', 'dictionary', 'hybrid', 'original_e8'):
            counts['raw_groups'] = int(stats.get('mode_counts', {}).get('0', 0))
        stats['registry_mechanism_counts'] = counts
        stats['registry_execution'] = 'cpu_reference'
        stats['serialized_bytes'] = len(value.data)
        return RegistryRecord(bytes(value.data), stats)

    def encode(self, words, layer, kind):
        words = np.asarray(words)
        if words.dtype != np.uint16 or words.ndim != 2:
            raise ValueError('expected uint16 [tokens,KV_heads*head_dim]')
        return self._record(self._encode(words, layer, kind))

    def decode(self, blob, layer, kind):
        blob = blob.data if hasattr(blob, 'data') else bytes(blob)
        return self._decode(blob, layer, kind)

    def encode_from_parent(self, parent_record):
        """Wrap exactly the supplied inner bytes; avoids duplicate encoding."""
        if self.parent is None:
            raise ValueError('this codec is not a whole-record wrapper')
        from kv_stacked import wrapper
        encoded = wrapper.wrap(parent_record, 6)
        if wrapper.unwrap(encoded.data) != parent_record.data:
            raise AssertionError('wrapper changed inner bytes')
        result = self._record(encoded)
        result.stats['inner_mechanism_counts'] = dict(parent_record.stats.get('registry_mechanism_counts', {}))
        result.stats['wrapper_inner_byte_identity'] = True
        return result


def build_registry(preset='main', artifacts=None, workspace='.', names=None, grid=False,
                   expected_model_identity=None):
    """Build adapters using ONLY caller-supplied target-model calibration.

    State files are created/read under workspace; complete serialized files
    count once per cache. Runtime arrays and temporary working space are extra.
    """
    from kv_smoke import codec as simple
    from kv_stacked import wrapper, field_codec, huffman_codec
    from kv_research import learned_codec, e8_codec
    from kv_residual import learned, learned_e8, direct
    from kv_coset_fields import combined, controls, temporal, learned_lower
    from kv_combinations import hybrid
    from kv_binary import binary_codec
    all_specs = _specs(grid)
    selected = candidate_specs(preset, grid) if names is None else {name: all_specs[name].copy() for name in names}
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    metadata = getattr(artifacts, 'metadata', {}) if artifacts is not None else {}
    if artifacts is not None and expected_model_identity is not None and metadata.get('model_identity') != expected_model_identity:
        raise ValueError('calibration model identity differs from the requested target model')
    models = getattr(artifacts, 'models', {}) if artifacts is not None else {}
    paths = getattr(artifacts, 'paths', {}) if artifacts is not None else {}
    built, states = {}, {}

    def model_info(key):
        if key not in models:
            raise ValueError(f'missing target-model calibration {key}; run calibration before testing')
        model = models[key]
        if (metadata.get('calibration_namespace') != 'qwen_kv_bench' or
                model.config.get('calibration_namespace') != 'qwen_kv_bench' or
                model.config.get('source_model_identity') != metadata.get('model_identity')):
            raise ValueError('model state lacks matching target-model calibration provenance')
        raw = learned_lower.model_bytes(model) if key.startswith('lower') else learned_codec.model_bytes(model)
        info = dict(model_key=key, serialized_bytes=len(raw), model_sha256=hashlib.sha256(raw).hexdigest(),
                    model_identity=metadata.get('model_identity'), calibration_namespace=metadata.get('calibration_namespace'))
        if key in paths:
            path = Path(paths[key]).resolve()
            if not path.is_relative_to(workspace):
                raise ValueError('calibrated model path must be inside workspace')
            if path.read_bytes() != raw:
                raise ValueError('calibrated model file differs from in-memory model')
            info['model_path'] = str(path)
        return model, info

    def build(name):
        if name in built:
            return built[name]
        spec = all_specs[name]
        family = spec['kind']
        info, state_size = {}, 0
        model = None
        if 'model' in spec:
            model, info = model_info(spec['model'])
            state_size = info['serialized_bytes']
        key = lambda layer, kind: f'layer_{layer:02d}_{kind}'
        parent = None
        if family == 'wrap':
            parent = build(spec['parent'])
            def enc(w, l, k):
                inner = parent.encode(w, l, k)
                wrapped = wrapper.wrap(inner, 6)
                if wrapper.unwrap(wrapped.data) != inner.data:
                    raise AssertionError('wrapper changed its parent record')
                return RegistryRecord(wrapped.data, {**wrapped.stats,
                    'inner_mechanism_counts': dict(inner.stats['registry_mechanism_counts']),
                    'wrapper_inner_byte_identity': True})
            dec = lambda b, l, k: parent.decode(wrapper.unwrap(b), l, k)
            info, state_size = dict(parent.state_info), parent.state_bytes
        elif family == 'raw':
            enc = lambda w, l, k: wrapper.raw_matrix(w, spec['shuffle'])
            dec = lambda b, l, k: wrapper.decode_raw_matrix(b)
        elif family == 'simple':
            enc = lambda w, l, k: simple.encode_matrix(w, spec['method'], group_size=64)
            dec = lambda b, l, k: simple.decode_matrix(b)
        elif family == 'field':
            enc = lambda w, l, k: field_codec.encode_matrix(w, strategy=spec['strategy'], level=6, group_size=64)
            dec = lambda b, l, k: field_codec.decode_matrix(b)
        elif family == 'huffman':
            enc = lambda w, l, k: huffman_codec.encode_matrix(w)
            dec = lambda b, l, k: huffman_codec.decode_matrix(b)
        elif family in ('lower_control', 'temporal', 'lower_learned'):
            def lower_functions(layer, kind):
                if family == 'lower_control':
                    return (lambda lo, e: controls.encode_lower(lo, e, spec['strategy']),
                            lambda b, e: controls.decode_lower(b, e, spec['strategy']))
                if family == 'temporal':
                    return (lambda lo, e: temporal.encode_lower(lo, e, predictor=spec['predictor']), temporal.decode_lower)
                return (lambda lo, e: learned_lower.encode_lower(lo, e, model, key(layer, kind), force_cosets=spec['force']),
                        lambda b, e: learned_lower.decode_lower(b, e, model, key(layer, kind)))
            mode = 3 if family == 'lower_learned' else 2
            def enc(w, l, k):
                lower_enc, lower_dec = lower_functions(l, k)
                return combined.encode_matrix(w, lower_enc, mode, lower_dec,
                    allow_expansion=family == 'lower_learned' and spec['force'])
            def dec(b, l, k):
                return combined.decode_matrix(b, lower_functions(l, k)[1], mode)
        elif family == 'dictionary':
            enc = lambda w, l, k: learned_codec.encode_matrix(w, model, key(l, k), group_size=64)
            dec = lambda b, l, k: learned_codec.decode_matrix(b, model)
        elif family == 'hybrid':
            enc = lambda w, l, k: hybrid.encode_matrix(w,
                learned_codec.encode_matrix(w, model, key(l, k), group_size=64),
                simple.encode_matrix(w, 'exponent', group_size=64))
            dec = lambda b, l, k: hybrid.decode_matrix(b, model)
        elif family == 'hybrid_tile':
            # The earlier coarse selector: one extra mode byte plus a complete
            # learned/exponent/raw/E8 record. Dictionary state still counts.
            def enc(w, l, k):
                candidates = [learned_codec.encode_matrix(w, model, key(l, k), 64),
                    simple.encode_matrix(w, 'exponent', 64), simple.encode_matrix(w, 'raw', 64),
                    e8_codec.encode_matrix(w, method='e8_symbol', symbol_step_log2=6)]
                selected_index = min(range(4), key=lambda i: len(candidates[i].data))
                selected_record = candidates[selected_index]
                data = bytes([selected_index]) + selected_record.data
                return RegistryRecord(data, dict(component_bytes={'tile_selector': 1,
                    'selected_complete_record': len(selected_record.data)},
                    tile_choice_counts={('learned', 'exponent', 'raw', 'e8')[selected_index]: 1}))
            def dec(b, l, k):
                if not b or b[0] > 3:
                    raise ValueError('invalid coarse hybrid selector')
                if b[0] == 0:
                    return learned_codec.decode_matrix(b[1:], model)
                if b[0] in (1, 2):
                    return simple.decode_matrix(b[1:])
                return e8_codec.decode_matrix(b[1:])
        elif family == 'original_e8':
            enc = lambda w, l, k: e8_codec.encode_matrix(w, method=spec['method'], group_size=64,
                symbol_step_log2=6, float_scale_multiplier=0.015625)
            dec = lambda b, l, k: e8_codec.decode_matrix(b)
        elif family == 'direct':
            enc = lambda w, l, k: direct.encode_matrix(w, strategy='best')
            dec = lambda b, l, k: direct.decode_matrix(b)
        elif family == 'residual':
            enc = lambda w, l, k: learned.encode_matrix(w, model, key(l, k),
                correction=spec['correction'], assignment=spec['assignment'], force=True)
            dec = lambda b, l, k: learned.decode_matrix(b, model)
        elif family == 'residual_e8':
            enc = lambda w, l, k: learned_e8.encode_matrix(w, model, key(l, k),
                step_log2=spec['step_log2'], correction=spec['correction'], family=spec['family'],
                assignment='radius', force=True, group_size=max(8, ((w.size + 7) // 8) * 8))
            dec = lambda b, l, k: learned_e8.decode_matrix(b, model)
        elif family == 'binary':
            state_key = spec['model'], spec['dimension'], spec['transform']
            if state_key not in states:
                state = binary_codec.make_state(model, dimension=spec['dimension'], transform=spec['transform'], seed=104729)
                path = workspace / 'binary_states' / (f"{spec['transform']}_d{spec['dimension']}_" + model.model_id[:16] + '.bin')
                expected = binary_codec.state_bytes(state)
                if path.exists() and path.read_bytes() != expected:
                    raise ValueError('existing binary state differs from calibrated target model')
                binary_codec.save_state(state, path)
                states[state_key] = state, path, expected
            state, path, raw_state = states[state_key]
            info = {**info, 'transform_path': str(path), 'transform_sha256': hashlib.sha256(raw_state).hexdigest(),
                'transform_bytes': len(raw_state), 'dictionary_bytes': state_size,
                'runtime_rotation_array_bytes': state.runtime_rotation_array_bytes,
                'dimension': state.dimension, 'transform': state.transform}
            state_size += len(raw_state)
            info['serialized_bytes'] = state_size
            enc = lambda w, l, k: binary_codec.encode_matrix(w, model, state, key(l, k),
                assignment='radius', force=True, amplitude_precision=spec['amplitude_precision'])
            dec = lambda b, l, k: binary_codec.decode_matrix(b, model, state)
        else:
            raise ValueError('unknown codec family ' + family)
        traits = dict(family=family, execution='cpu_reference', gpu_native=False, exact=True,
            model_specific=state_size > 0, options=spec.copy(),
            fallback_policy='explicit in registry_mechanism_counts and original codec diagnostics',
            whole_record_grouping=family == 'residual_e8')
        adapter = CodecAdapter(name, enc, dec, state_size, info, traits, parent)
        built[name] = adapter
        return adapter
    return {name: build(name) for name in selected}
