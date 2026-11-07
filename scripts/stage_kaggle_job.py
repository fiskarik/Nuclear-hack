"""Prepare a private Kaggle dataset and a reproducible, larger adaptive GPU job."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import nbformat


def stage(archive,checkpoint,output,username='fiskarik',dataset_slug='nuclear-hack-doe-training-v3'):
    root=Path(__file__).resolve().parents[1]
    archive,checkpoint,output=Path(archive),Path(checkpoint),Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError(output)
    dataset=output/'dataset';notebook=output/'notebook'
    dataset.mkdir(parents=True);notebook.mkdir()
    shutil.copy2(archive,dataset/'sbs-textures-128.zip')
    shutil.copy2(checkpoint,dataset/'adaptive-parent.pt')
    metadata={'title':'Nuclear Hack DOE Training V3','id':f'{username}/{dataset_slug}',
              'licenses':[{'name':'other'}],
              'description':'Private inputs: Screaming Brain Studios textures carry CC0/Public Domain; model weights remain copyright of the repository owner and are staged for private training only.'}
    (dataset/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (dataset/'README.md').write_text('Private training inputs. SBS texture archive includes its CC0/Public Domain License.txt. '
        'Model weights belong to the repository owner; no new public license is granted by staging. '
        'Use datasets create without --public.\n')
    nb=nbformat.read(root/'notebooks/fresnel_doe_kaggle.ipynb',as_version=4)
    code=[c for c in nb.cells if c.cell_type=='code']
    code[0].source+='''
# Resolve only the files from the attached private dataset; credentials stay outside the notebook.
def unique_input(filename):
    paths = list(Path('/kaggle/input').rglob(filename))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one {filename}, found {len(paths)}')
    return str(paths[0])
if os.environ.get('DOE_QUICK_CHECK') != '1':
    os.environ['DOE_TEXTURE_ZIP'] = unique_input('sbs-textures-128.zip')
    os.environ['DOE_INITIAL_CHECKPOINT'] = unique_input('adaptive-parent.pt')
os.environ['DOE_MODEL_KIND'] = 'adaptive'
'''
    code[1].source+='''
# Larger GPU experiment; no claim that 0.8 is guaranteed.
config.objective = 'perceptual'
if not GPU and not QUICK_CHECK:
    raise RuntimeError('This job requires an enabled Kaggle GPU; check account verification and quota')
if not QUICK_CHECK:
    counts = (2048, 32, 32)
    config.stages = max(24, config.stages)
    config.initializer_depth = 4
    config.epochs = 80
    config.batch_size = 8
    config.learning_rate = 0.0003
print('GPU experiment configuration:', config)
'''
    nb.cells.insert(1,nbformat.v4.new_markdown_cell('''## Подготовленный GPU-эксперимент

Приватный входной набор: SBS ZIP и родительские веса. 24 обучаемых оптических слоя,
80 эпох, 2048 train-четвёрок. Validation/test остаются по 32 четвёрки и не используются
как обучающие данные. Требуется Kaggle GPU и доступная квота. Цель 0.8 пока не
подтверждена; этот запуск предназначен для дальнейшего исследования.
'''))
    nbformat.validate(nb)
    nbformat.write(nb,notebook/'adaptive_gpu.ipynb')
    kernel={'id':f'{username}/nuclear-hack-adaptive-gpu','title':'Nuclear Hack Adaptive GPU',
            'code_file':'adaptive_gpu.ipynb','language':'python','kernel_type':'notebook','is_private':True,
            'enable_gpu':True,'enable_internet':False,'dataset_sources':[f'{username}/{dataset_slug}'],
            'competition_sources':[],'kernel_sources':[],'model_sources':[]}
    (notebook/'kernel-metadata.json').write_text(json.dumps(kernel,indent=2)+'\n')
    manifest={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
              'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'dataset':metadata['id'],'kernel':kernel['id'],'privacy':'private; never add --public',
              'runtime_status':'not submitted; authentication and applied Kaggle network access required'}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output/'README.md').write_text(f'''# Prepared Kaggle GPU experiment

No remote submission has occurred. Keep the dataset and notebook private.
Use the existing cloud KAGGLE_API_TOKEN binding and HTTPS proxy. Never paste tokens into code or chat.
Run the following commands from this job directory.

1. Verify authentication: `kaggle kernels list --mine --page-size 1`.
2. Check `{metadata['id']}` is not an unrelated dataset. If unused, run
   `kaggle datasets create -p dataset` without `--public`.
3. Wait until the dataset is ready, then run
   `kaggle kernels push -p notebook --accelerator NvidiaTeslaT4 -t 21600`.
4. Inspect `kaggle kernels status {kernel['id']}`. After complete, download the output
   with `kaggle kernels output {kernel['id']} -p NEW_DIRECTORY` and inspect actual metrics.

Saved files and a successful upload are not evidence of a completed GPU training run.
''')
    print(json.dumps(manifest,indent=2));return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('archive','checkpoint','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--username',default='fiskarik');p.add_argument('--dataset-slug',default='nuclear-hack-doe-training-v3')
    stage(**vars(p.parse_args()))
