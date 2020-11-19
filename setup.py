from distutils.core import setup

setup(name='librl',
      version='0.1.1',
      description='Library for various DRL and DL projects.',
      author='Matthew McRaven',
      author_email='mkm302@georgetown.edu',
      url='https://github.com/Matthew-McRaven/librl',
      packages=['librl',
      'librl.agent',
      'librl.nn',
      'librl.replay',  
      'librl.task',
      'librl.train', 'librl.train.cc']
     )