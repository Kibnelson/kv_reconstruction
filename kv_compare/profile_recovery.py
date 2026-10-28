"""Narrow, auditable repair of the published partial profile run."""
import copy
import hashlib
import json
import os
from pathlib import Path

LEGACY_CODE={
    "scripts/run_llama_profile_baselines.py": "2a6cdbb98b289d4c82cde7646480fd43c5761fbda95e38df406de065acbb0b36",
    "scripts/run_llama_profile_a4500.sh": "effb4022391f2158fab1b6ca300bdd4d528835220a224ec2fee96b43571cfda9",
    "scripts/package_llama_profile.py": "36624c25ba469a40d863243628a6099a277b1c718d9bc70240f29566cd9355f9",
    "scripts/setup_qwen_cuda_toolchain.py": "c6f383bdf3026e1af6261d878762ca3cb6415a020c33606bff96b0fc7e5c9f39",
    "kv_compare/causal_profile.py": "8ab17bff0ce1bed20098b6746e5638f964326d4c1c6f3480c871ce2606dbf2f9",
    "kv_compare/official_exans.py": "df2134b5709b26cf631bf281b1a05225d9256277a3e4c62f915e6985b4869185",
    "native/exans_build_compat.h": "a1c9a5b2c6621b6f929a481250067e4618f8dd3d46084fa1be1a783dc2a70182",
    "scripts/bench_llama_baselines.py": "bd3cbde7a9ec8dbaf296377fbcf0369761911f117cae6663fb63f04b7a740924",
    "kv_compare/framed_baselines.py": "7719b5cb246791be1e4d73921b95515d2207b6b0fdc6a1292bdf1ce243467c27",
    "scripts/package_llama_causal.py": "ee1e6ae032de9761ee0f41f4cce663059c17c0abad5d0609bd88a13cacb7380b"
}


RECOVERY_CODE_V1={
    "scripts/run_llama_profile_baselines.py": "ae34a763aef1abd81082004d73f45bb88bdfc86f21ef2c677f2b1fa7f2d857f0",
    "scripts/run_llama_profile_a4500.sh": "effb4022391f2158fab1b6ca300bdd4d528835220a224ec2fee96b43571cfda9",
    "scripts/package_llama_profile.py": "9c5c186100659d7d12f8e6ab7f8f265b4126b1a94ce795aed975f5c2124c90c0",
    "scripts/setup_qwen_cuda_toolchain.py": "c6f383bdf3026e1af6261d878762ca3cb6415a020c33606bff96b0fc7e5c9f39",
    "kv_compare/causal_profile.py": "8ab17bff0ce1bed20098b6746e5638f964326d4c1c6f3480c871ce2606dbf2f9",
    "kv_compare/official_exans.py": "df2134b5709b26cf631bf281b1a05225d9256277a3e4c62f915e6985b4869185",
    "native/exans_build_compat.h": "a1c9a5b2c6621b6f929a481250067e4618f8dd3d46084fa1be1a783dc2a70182",
    "scripts/bench_llama_baselines.py": "bd3cbde7a9ec8dbaf296377fbcf0369761911f117cae6663fb63f04b7a740924",
    "kv_compare/framed_baselines.py": "7719b5cb246791be1e4d73921b95515d2207b6b0fdc6a1292bdf1ce243467c27",
    "scripts/package_llama_causal.py": "ee1e6ae032de9761ee0f41f4cce663059c17c0abad5d0609bd88a13cacb7380b",
    "kv_compare/profile_recovery.py": "fb7acf7cbe087f628194281371d72a92fb73fc721faaf77237d63c02d1ea89f3",
    "scripts/prepare_aqua_calibration.py": "a3f32784b1d6c4be67f20d32265bfe4c9d9598bfd9ad1196c2236cc711b52e16",
    "benchmark_fixtures/aqua_redpajama_backup_manifest.json": "4caecc08c02a2ce96590eba1c8751e5eef4806f81d10c107670373309a462d5e"
}


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''): h.update(block)
    return h.hexdigest()


def identity(recipe):
    return hashlib.sha256(json.dumps(recipe,sort_keys=True).encode()).hexdigest()


def runtime_environment(env,home,extra=()):
    """Locate runtime DSOs for standalone author executables as well as nvcc."""
    env=dict(env); home=Path(home)
    paths=[home/'lib64',home/'lib',*map(Path,extra)]
    original=[p for p in env.get('LD_LIBRARY_PATH','').split(os.pathsep) if p]
    values=[str(p) for p in paths if p.is_dir()]+original
    env['LD_LIBRARY_PATH']=os.pathsep.join(dict.fromkeys(values))
    env.update(CUDA_HOME=str(home),CUDA_PATH=str(home),PATH=str(home/'bin')+os.pathsep+env.get('PATH',''))
    return env


def recover_partial(old,recipe,out):
    """Only known runner releases may migrate; all completed evidence is kept.

    Changes repair runner/loader/calibration preparation, never the measured core.
    Backup calibration is explicit and allowed only before training has completed.
    """
    out=Path(out).resolve(); before=old['recipe']
    legacy=before.get('code')==LEGACY_CODE
    recovery_v1=before.get('code')==RECOVERY_CODE_V1
    source_choice_only=(before.get('code')==recipe.get('code') and
        before.get('aqua_calibration','original')=='original' and recipe.get('aqua_calibration')=='backup')
    if old['identity']!=identity(before) or not (legacy or recovery_v1 or source_choice_only):
        raise ValueError('resume identity changed outside the supported repair; existing run preserved')
    allowed={'profile','toolchain','aqua_setup','aqua_raw'}
    if recovery_v1: allowed.add('exans_build')
    if source_choice_only: allowed.update({'exans_build','exans'})
    completed={name for name,s in old['stages'].items() if s.get('status')=='completed'}
    if not completed<=allowed or old.get('status') not in ('partial','failed','running'):
        raise ValueError('this repair cannot migrate completed baseline training/measurements')
    a={k:v for k,v in before.items() if k not in ('code','aqua_calibration')}
    b={k:v for k,v in recipe.items() if k not in ('code','aqua_calibration')}
    # Stored JSON uses lists for source pins; the runner constructs tuples.
    if json.dumps(a,sort_keys=True)!=json.dumps(b,sort_keys=True):
        raise ValueError('input, model, mode or official protocol changed; existing run preserved')
    for name in completed:
        for rel,expected in old['stages'][name].get('artifacts',{}).items():
            path=(out/rel).resolve()
            if not path.is_relative_to(out) or digest(path)!=expected:
                raise ValueError('completed evidence changed before recovery: '+rel)
    previous=out/'results.json'; archive=out/'resume_history'/('results_'+old['identity']+'.json')
    archive.parent.mkdir(parents=True,exist_ok=True)
    if archive.exists():
        if archive.read_bytes()!=previous.read_bytes():
            raise ValueError('recovery archive already differs; preserve existing evidence')
    else: archive.write_bytes(previous.read_bytes())
    result=copy.deepcopy(old)
    result.setdefault('resume_history',[]).append(dict(previous_identity=old['identity'],
        previous_recipe=before,archived_result=str(archive.relative_to(out)),archived_sha256=digest(archive),
        preserved_completed_stages=sorted(completed),
        reason='checkpoint-preserving runtime/calibration compatibility repair; completed artifacts unchanged'))
    result.update(recipe=recipe,identity=identity(recipe),status='running')
    return result
