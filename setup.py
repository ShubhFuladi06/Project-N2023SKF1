from setuptools import setup, find_packages, setup 

HYPEN_E_DOT='-e .'
def get_requirements(file_path: str) -> list[str]:                    ## the file name must be string, and the return type is list of strings.
    
    """Read the requirements from a file and return them as a list."""
    requirements=[]
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
            
    return requirements



setup(
    name='my_ml_package',
    version='0.1.0',
    author='SKF',
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt')
)