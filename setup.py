import numpy

from setuptools import setup
from Cython.Build import cythonize

setup(
    name='MachineLearningForImageAnalysis',
    version='0.1',
    packages=['MLIA'],
    url='',
    license='MIT',
    author='Ivar Stangeby',
    author_email='',
    description='',
    ext_modules=cythonize('**/*.pyx', annotate=True, build_dir="MLIA/src_c/build"),
    include_dirs=[numpy.get_include()]
)
