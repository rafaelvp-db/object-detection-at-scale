from setuptools import setup, find_packages

setup(
    name='odd',
    version='0.1.0',
    packages=find_packages(
        include=['odd', 'odd.*']
    )
)