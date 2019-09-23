from setuptools import find_packages, setup
from pathlib import Path


def set_init_dir():
    symlink_dict = {
        'lib/local_lib': 'src'
    }

    for sym_name, tgt_name in symlink_dict.items():
        sym_p = Path(sym_name)
        tgt_p = Path(tgt_name)
        if sym_p.exists() or not tgt_p.exists():
            continue
        for i in range(len(sym_p.parts) - 1):
            tgt_p = Path('../').joinpath(tgt_p)
        sym_p.symlink_to(tgt_p)


set_init_dir()
setup(
    name='local-lib',
    version='0.0.1',
    packages=find_packages('lib'),
    package_dir={'': 'lib'}
)
