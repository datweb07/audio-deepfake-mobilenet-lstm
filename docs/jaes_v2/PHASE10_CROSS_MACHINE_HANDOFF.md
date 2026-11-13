# Phase 10 cross-machine handoff

## What is on GitHub

Read-only remote verification with `git ls-remote origin refs/heads/main` returned `8b3c6cb08052c7569e77a60fe26dcf333abf651c`, matching the checked V5 commit. The remote has source, configurations, requirements and V5. The repository-wide `outputs/` ignore excludes the Phase 8 protocol, medium manifest and selection rows, candidate 1,280 manifest, qualification comparator and both pilot trees. A clone or GitHub ZIP plus an existing ASVspoof archive is insufficient. Neither the V6 corrections nor this handoff were published in this task.

Byte comparison also found seven tracked files with different bytes between the Git object and the local checkout. Four concern the qualified bundle directly: `config.py`, `src/model.py`, the prospective registry and the template JSON. The scientific launcher has a byte-level release gate. Blanket newline conversion would change frozen fingerprints. The companion ZIP carries exact qualified bytes and the runtime metadata omitted by Git. It was read back entry-by-entry and verified against the source. The combination of Git blobs plus ZIP overlay was simulated by hash. It has not been run on the friend's machine.

## Files to send privately

1. [phase10_640_preflight_handoff.zip](../../outputs/jaes_v2/final_supervisor_closure/phase10_640_preflight_handoff.zip), 208,003,712 bytes. SHA-256 `5e855ea7854a35da65d0362603ba63029bfc5aa0e63498416719cdd556cc3dc5`. Includes 90 Git-ignored runtime files, candidate metadata, the exact byte overlays and `phase10_handoff_hashes.json`. Pilot checkpoints serve the release admission check.
2. This handoff document. The [Git audit JSON](../../outputs/jaes_v2/final_supervisor_closure/github_handoff_audit.json) lists all member hashes, version pins and checkout differences.
3. Exact FFmpeg build and ImageNet cache through the appropriate distribution or authorized private copy. They are outside the ZIP. FFmpeg must resolve first on PATH and hash to `74db6c184a03dba2bdfe23e1a1f41cf5a8385bc1de6a7a1b26db1dc541abef93` (observed FFmpeg 8.0.1 Gyan full build). The MobileNetV3Small cache is `weights_mobilenet_v3_small_224_1.0_float_no_top_v2.h5` with SHA-256 `002803a73df2ab65425267ee2082d1ec213b6aa9716f19ac1a1ed5361239c5f6` in the default Keras models directory. Cache is relevant to model initialization, while FFmpeg identity is checked at preflight.

The ZIP has no raw audio, evaluation lock approvals, .venv, historical scientific checkpoints, six-run score trees or credentials. Manifests carry the absolute project path by design. Keep them private. Leave the existing .gitignore in place.

## Read-only preflight on the second Windows machine

This exercise checks setup for the already released 640 profile. Neither 9307 retraining nor 1,280 training is requested. Use a fresh clone on the exact path encoded in the sealed manifests. Windows paths are checked by the loader. If D: or this root is unavailable, stop and arrange a separately qualified portability path. Rewriting a path in a sealed manifest changes its byte hash.

```powershell
git clone -c core.autocrlf=false https://github.com/datweb07/audio-deepfake-mobilenet-lstm.git D:\audio-deepfake-mobilenet-lstm
Set-Location D:\audio-deepfake-mobilenet-lstm
git checkout --detach 8b3c6cb08052c7569e77a60fe26dcf333abf651c
Get-FileHash 'C:\path\to\phase10_640_preflight_handoff.zip' -Algorithm SHA256
# After confirming the ZIP SHA above, overlay the exact qualified bytes.
Expand-Archive -LiteralPath 'C:\path\to\phase10_640_preflight_handoff.zip' -DestinationPath . -Force

$lock = Get-Content phase10_handoff_hashes.json -Raw | ConvertFrom-Json
foreach ($map in @($lock.entries, $lock.qualified_tracked_sha256)) {
  foreach ($entry in $map.PSObject.Properties) {
    if ((Get-FileHash -LiteralPath $entry.Name -Algorithm SHA256).Hash -ine $entry.Value) {
      throw "Phase 10 byte mismatch: $($entry.Name)"
    }
  }
}
if ((Get-FileHash -LiteralPath (Get-Command ffmpeg).Source -Algorithm SHA256).Hash -ine $lock.ffmpeg_sha256) {
  throw 'Wrong FFmpeg executable on PATH'
}
```

Place existing ASVspoof 2019 LA train/dev directories under `D:\audio-deepfake-mobilenet-lstm\data\public\asvspoof2019_la_v1\LA\`. For example, `LA\ASVspoof2019_LA_train\flac\LA_T_*.flac` and `LA\ASVspoof2019_LA_dev\flac\LA_D_*.flac`. If starting from the original official LA ZIP, extract only those two directory members with their existing names. The 640 preflight verifies all required audio bytes. Official eval and calibration audio are not needed to run this preflight. Keep the original ASVspoof archive separately if it is already available.

Install Windows Python 3.11.9 and create a local virtual environment. The package list in the lock is the measured runtime, including the CPU TensorFlow distribution.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe --version
$runtime = $lock.packages.PSObject.Properties | ForEach-Object { "$($_.Name)==$($_.Value)" }
.\.venv\Scripts\python.exe -m pip install -r requirements.txt @runtime
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_experiment resources
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_erm_s9307_v2
.\.venv\Scripts\python.exe -B -X utf8 -m scripts.jaes_v2.phase10_medium preflight --experiment p10_medium_channel_s9307_v2
```

Return the two preflight logs and resource JSON. The launcher runs on CPU. Another GPU does not change the released backend or the required manifest admission. Neither these commands nor this ZIP constitute a qualified 1,280 profile. [Train-size qualification](TRAIN1280_FINAL_QUALIFICATION.md)
