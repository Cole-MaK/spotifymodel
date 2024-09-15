from setuptools import find_packages, setup
from typing import List

hyp_e_dot = '-e .'
def get_requirements(file_path:str) -> List[str]:
    '''
    This function returns the list of requirements
    '''
    requirements = []
    with open(file_path) as file:
        requirements=file.readlines()
        requirements = [req.replace('\n','') for req in requirements]

        if hyp_e_dot in requirements:
            requirements.remove(hyp_e_dot)
    
    return requirements


setup(
name = 'song genre prediction',
version = '0.0.1',
author = 'Cole',
author_email = 'mrcolemak@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')
)