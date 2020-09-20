import codecs
import os.path

from setuptools import setup, find_packages

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

setup(
    name="pylce",
    version=get_version("pylce/version.py"),
    author="Cong Liang",
    author_email="cong.liang@tju.edu.cn",
    description="level of correlated evolution estimation",
    packages=find_packages()
)
