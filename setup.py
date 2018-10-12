import re
from setuptools import setup, find_packages

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

try:
    requirements = parse_requirements('requirements.txt')
except OSError:
	requirements = []

setup(
    name='ab_eval',
    version=0.1,
    description="Experiment Analysis Library",
    author="Yiannis Moschonas",
    author_email='yiannis.moschonas@deliveryhero.com',
    url='https://github.com/YiannisDH/ab_eval',
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='ab_eval',
    #test_suite='tests',
    #tests_require=test_requirements
)