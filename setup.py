from setuptools import setup, find_packages

setup(
    name='dunl',
    version='0.1.0',
    packages=["src"],
    packages=find_packages(),
    install_requires=[
        'configmypy==0.1.0',
        'h5py==3.10.0',
        'hillfit==0.1.7',
        'matplotlib',
        'numpy>=2.2.4',
        'scikit_learn',
        'scipy==1.15.2',
        'tensorboardX==2.6.2.2',
        'torch==2.1.2',
        'torchvision>=0.6.2',
        'tqdm==4.66.1',
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