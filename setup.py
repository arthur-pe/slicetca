from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='slicetca',
    packages=['slicetca'],
    version='0.1.5',

    description='Package to perform Slice Tensor Component Analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/arthur-pe/slicetca',
    author='Arthur Pellegrino',
    license='MIT',
    install_requires=['torch',
                      'numpy',
                      'matplotlib',
                      'tqdm'
                      ],
    python_requires='>=3.8',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Mathematics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
)