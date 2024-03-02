import setuptools as tools

tools.setup(
    name="multiflow",
    packages=[
        'openfold',
        'multiflow',
        'ProteinMPNN'
    ],
    package_dir={
        'openfold': './openfold',
        'multiflow': './multiflow',
        'ProteinMPNN': './ProteinMPNN',
    },
)
