#!/usr/bin/env python3
"""Explicit, pinned backup calibration via AQUA's supported local-token input.

This never claims the backup's row order is certified against the removed
original. It follows the released sampling rule, 256x8192 recipe and author model code.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_llama_causal import sha


def tokenizer_plan(model_path):
    """The pinned Llama-3 checkpoint provides a fast-only tokenizer JSON.

    Its generic PreTrainedTokenizer config has no implemented slow vocabulary.
    Use the checkpoint backend used by the author's successful PPL entry point.
    """
    path=Path(model_path)/'tokenizer_config.json'
    config=json.loads(path.read_text()) if path.exists() else {}
    generic=config.get('tokenizer_class') in ('PreTrainedTokenizer','PreTrainedTokenizerFast')
    if generic and not (Path(model_path)/'tokenizer.json').is_file():
        raise ValueError('fast-only checkpoint is missing tokenizer.json')
    return dict(use_fast=generic,reason='checkpoint_fast_backend_as_official_ppl' if generic else 'author_default_slow',
        tokenizer_config_sha256=sha(path) if path.exists() else None,
        tokenizer_json_sha256=sha(Path(model_path)/'tokenizer.json') if generic else None)


def sample_records(data,tokenizer,*,nsamples=256,seqlen=8192,seed=0):
    """Released get_red_pajama's sampling rule, with an explicit data source."""
    import torch
    rng=random.Random(seed)
    tokenizer.bos_token_id=1; tokenizer.eos_token_id=2
    batches=[]; selections=[]
    for _ in range(nsamples):
        while True:
            index=rng.randint(0,len(data)-1)
            text=data[index]['text']
            tokens=tokenizer(text,return_tensors='pt').input_ids
            if tokens.shape[1]>seqlen: break
        start=rng.randint(0,tokens.shape[1]-seqlen-1)
        batch=tokens[:,start:start+seqlen].clone()
        if batch.dtype!=torch.int64 or batch.shape!=(1,seqlen):
            raise ValueError('invalid calibration token geometry')
        batches.append(batch)
        selections.append(dict(row=index,start=start,text_sha256=hashlib.sha256(text.encode()).hexdigest(),
            token_sha256=hashlib.sha256(batch.numpy().astype('<i8').tobytes()).hexdigest()))
    return batches,selections


def prepare(source,model_path,output,receipt,cache,manifest_path):
    import torch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    manifest=json.loads(manifest_path.read_text())
    plan=tokenizer_plan(model_path)
    recipe=dict(manifest_sha256=sha(manifest_path),sampler_source_sha256=sha(source/'aquakv/datautils.py'),
        model_snapshot=str(model_path),seed=0,nsamples=256,seqlen=8192,tokenizer=plan)
    if receipt.exists() and output.exists():
        old=json.loads(receipt.read_text())
        if old['recipe']==recipe and sha(output)==old['token_file_sha256']:
            return old
        raise ValueError('existing calibration tokens/provenance changed; preserve them and choose a new output')
    # Validate the checkpoint backend before any dataset download/preparation.
    tokenizer=AutoTokenizer.from_pretrained(model_path,use_fast=plan['use_fast'],trust_remote_code=None,local_files_only=True)
    if plan['use_fast'] and not tokenizer.is_fast:
        raise ValueError('the checkpoint fast tokenizer backend did not load')
    paths=[]
    for e in manifest['files']:
        path=Path(hf_hub_download(repo_id=manifest['repo'],repo_type='dataset',revision=manifest['revision'],
            filename=e['path'],cache_dir=cache))
        if path.stat().st_size!=e['bytes'] or sha(path)!=e['sha256']:
            raise ValueError('pinned backup shard checksum differs: '+e['path'])
        paths.append(str(path))
    data=load_dataset('parquet',data_files={'train':paths},split='train',cache_dir=str(cache/'arrow'))
    if len(data)!=manifest['rows'] or not {'text','meta'}<=set(data.column_names):
        raise ValueError('backup is missing expected rows/columns; no subset fallback')
    batches,selections=sample_records(data,tokenizer)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.part'); torch.save(batches,temporary); temporary.replace(output)
    checked=torch.load(output,weights_only=True)
    if len(checked)!=256 or any(x.shape!=(1,8192) or x.dtype!=torch.int64 for x in checked):
        raise ValueError('saved calibration token validation failed')
    result=dict(recipe=recipe,source=manifest,token_file_sha256=sha(output),dataset_fingerprint=data._fingerprint,
        rows=len(data),selections=selections,source_mode='explicit_pinned_backup',
        equivalence_to_original='unverified; do not label as exact original-data reproduction',
        method='unchanged author train_predictors.py via its supported local pretokenized-data argument')
    receipt.write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--model-path',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=ROOT/'benchmark_fixtures/aqua_redpajama_backup_manifest.json')
    a=p.parse_args()
    result=prepare(a.source,a.model_path,a.output,a.receipt,a.cache,a.manifest)
    print('Prepared 256x8192 calibration tokens from explicit pinned backup:',result['token_file_sha256'])


if __name__=='__main__': main()
