from setuptools import setup

setup(
    name='dunl',
    version='0.1.0',
    packages=['dunl'],
    install_requires=[
        'configmypy',
        'h5py',
        'hillfit',
        'matplotlib',
        'numpy>=2.2.4',
        'scikit_learn',
        'scipy',
        'tensorboardX',
        'torch>=2.1.2',
        'torchvision>=0.6.2',
        'tqdm',
    ],
    author='Bahareh Tolooshams',
    author_email='btolooshams@gmail.com',
    description='DUNL for Computational Neuroscience (published at Neuron in 2025)',
    url='https://github.com/btolooshams/dunl-compneuro',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)