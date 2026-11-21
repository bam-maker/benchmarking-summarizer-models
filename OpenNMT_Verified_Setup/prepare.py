"""Download the verified official checkpoints and patch a pinned OpenNMT checkout."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parent
COMMIT = '6db7ec1b3afae28c257a0416dd947022c3b12bc0'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    legacy = ROOT / 'legacy'
    if not legacy.exists():
        subprocess.run(['git', 'clone', '--depth', '1', '--branch', '0.2.1',
                        'https://github.com/OpenNMT/OpenNMT-py.git', str(legacy)], check=True)
    actual = subprocess.check_output(['git', '-C', str(legacy), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError('Unexpected OpenNMT revision; use a fresh kit directory.')
    patches = {
        'onmt/modules/global_attention.py': ('1 - mask,', '~mask.bool(),'),
        'onmt/translate/beam.py': ('best_scores_id / num_words', 'best_scores_id // num_words'),
    }
    for relative, (old, new) in patches.items():
        path = legacy / relative
        content = path.read_text()
        if old in content:
            path.write_text(content.replace(old, new))
        elif new not in content:
            raise RuntimeError('Unexpected source in ' + relative)
    manifest = json.loads((ROOT / 'checkpoint_manifest.json').read_text())
    for name, item in manifest.items():
        target = ROOT / (name + '.pt')
        if not target.exists() or digest(target) != item['sha256']:
            temporary = target.with_suffix('.download')
            print('Downloading', name, item['bytes'], 'bytes', flush=True)
            urllib.request.urlretrieve(item['url'], temporary)
            if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
                temporary.unlink(missing_ok=True)
                raise RuntimeError('Checkpoint verification failed for ' + name)
            temporary.replace(target)
        print('Verified', name, flush=True)
    subprocess.run([sys.executable, str(ROOT / 'convert_checkpoints.py')], check=True)

if __name__ == '__main__':
    main()
