from setuptools import setup 
  
setup( 
    name='apc', 
    version='0.1', 
    description='Package for the dynamics abstraction project', 
    author='Vishwas Sathish', 
    author_email='vsathish@cs.washington.edu', 
    packages=['dataloaders', 'environments', 'models'], 
    install_requires=[ 
        'numpy', 
        'pandas', 
    ], 
) 