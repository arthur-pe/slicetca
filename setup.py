from setuptools import setup

setup(
    name='slicetca',
    version='0.1.1',
    description='Package to perform Slice Tensor Component Analysis',
    url='https://github.com/arthur-pe/slicetca',
    author='Arthur Pellegrino',
    license='MIT',
    packages=['slicetca'],
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