from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description=(
        'Tweet Emotion Classification — an end-to-end MLOps project '
        'using MLflow and DagsHub for experiment tracking.'
    ),
    author='Deepu',
    license='MIT',
    python_requires='>=3.8',
)
