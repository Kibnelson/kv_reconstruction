# First push to the empty GitHub repository

The code in this bundle is ready to publish to `Kibnelson/kv_reconstruction`. The ChatGPT GitHub connection returned HTTP 403, `Resource not accessible by integration`, when creating the first README; no files were written to GitHub.

## Push from your machine

Download this archive into your working directory, then extract it. The ZIP contains a `kv_reconstruction/` directory and excludes git metadata, environments, model weights, caches, and generated run results.

```bash
unzip kv_reconstruction_repo.zip
cd kv_reconstruction
git init -b main
git add .
git commit -m "Add exact KV benchmarks and modern Transformers cache support"
git remote add origin https://github.com/Kibnelson/kv_reconstruction.git
git push -u origin main
```

Use your existing GitHub Git authentication. An HTTPS remote requires a supported Git credential method such as GitHub CLI/credential manager/token authentication; an account password is not used for Git pushes. If your machine already uses SSH, you can use `git@github.com:Kibnelson/kv_reconstruction.git` as the origin instead. Keep credentials on your own machine.

The repository was empty when checked. If someone adds commits before you push, inspect and integrate them rather than force-pushing. If Git asks for your commit identity, configure your own name and email before retrying the commit.

After the first push, future checkouts can use the clone and update commands in README.md.

## Enable writes from the GitHub connection

Open [GitHub's installed-app settings](https://github.com/settings/installations). Ensure the app used by the connection is installed on your account and has access to `kv_reconstruction`. Review any requested permission updates. Writing code requires the Contents repository permission with write access; updating the included Actions workflow also requires workflow write access. You can only grant permissions the app itself requests; a read-only app cannot be elevated by selecting more repositories.

If the connection still cannot obtain those permissions, push using your own Git authentication as above. No changes to the benchmark code or NVIDIA driver are required for this access problem.

## The runtime fix

This bundle includes the corrected `qwen_kv_gpu_bench.py`. It handles modern `cache.layers[i].keys/values` and legacy cache formats. Functional suites passed on Transformers 4.57.6 and 5.20.1, including random BF16 Qwen3 prefill, exact restoration, and continued decoding. GPU execution remains to be checked on your NUC.

```bash
python qwen_kv_gpu_bench.py --device cuda --preset smoke \
  --lengths 2048 --cases 1 --output runs/qwen_smoke_fixed
```
