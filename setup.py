from setuptools import setup, find_packages

setup(
    name='grund',
    version='0.0.1',
    packages=find_packages(),
    url='https://github.com/csxeba/grund',
    license='GPLv3',
    author='Csxeba',
    author_email='csxeba@gmail.com',
    description='My Reinforcement Learning playground',
    long_description=open("README.md").read(),
    extras_require={
        "full": ["keras>=2.0"]
    }
)
