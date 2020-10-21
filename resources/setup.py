from setuptools import setup

setup(
    name='cubert',
    version='0.0.1',
    packages=['cubert'],
    install_reuirements=[
        "bert-tensorflow==1.0.4",
        "dopamine-rl==3.0.1",
        "regex==2020.9.27",
        "tensor2tensor==1.15.7",
        "absl-py==0.10.0",
        "tensorflow==1.15"
    ],
    python_requires='>=3.7',

    classifiers=[
        'Framework :: IPython',
        'Framework :: Jupyter',
    ],
)

