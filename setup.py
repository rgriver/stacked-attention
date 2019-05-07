from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'tensorflow-gpu',
    'numpy']

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    include_package=True,
    install_requires=REQUIRED_PACKAGES,
    description='First trainer package',
    requires=[]
)
