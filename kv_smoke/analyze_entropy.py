"""Descriptive marginal entropies of the included actual BF16 cache sample."""
import json, pathlib
import numpy as np
root=pathlib.Path(__file__).parent/"results"
sample=np.load(root/"sample_raw_cache.npz")
def entropy(symbols,nvalues):
    counts=np.bincount(symbols.astype(np.int64),minlength=nvalues)
    prob=counts[counts>0]/symbols.size
    return float(-(prob*np.log2(prob)).sum())
results={}
for kind in ("K","V"):
    words=np.concatenate([sample[key].reshape(-1) for key in sample.files if key.endswith("_"+kind)])
    results[kind]={
      "values":int(words.size),
      "word_unigram_entropy_bits":entropy(words,65536),
      "sign_marginal_entropy_bits":entropy(words>>15,2),
      "exponent_marginal_entropy_bits":entropy((words>>7)&255,256),
      "mantissa_marginal_entropy_bits":entropy(words&127,128),
    }
record={"sample_tokens":128,"statistics":results,"interpretation":
 "Empirical marginal histograms of one synthetic-context model cache. These are descriptive, finite-sample estimates, not a bound on a conditional, learned, or predictive codec, nor measured achievable bitrates. Dictionary, finite-block, access and model costs are excluded from entropy estimates."}
(root/"entropy_diagnostic.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record,indent=2))

