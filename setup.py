"""Setup file for relic

   Adapted from the Python Packaging Authority template."""

import sys
import warnings
import re

from setuptools import setup, find_packages  # Always prefer setuptools
from codecs import open  # To use a consistent encoding
from os import path, walk

MAJOR = 0
MINOR = 0
MICRO = 1
ISRELEASED = False
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)
QUALIFIER = ''

DISTNAME = 'relic'
LICENSE = 'MIT'
AUTHOR = 'Matthias Dusch'
AUTHOR_EMAIL = 'matthias.dusch@uibk.ac.at'
URL = 'https://github.com/matthiasdusch/relic'
CLASSIFIERS = [
        # How mature is this project? Common values are
        # 3 - Alpha  4 - Beta  5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License' +
        'Programming Language :: Python :: 3.5',
    ]

DESCRIPTION = '...'
LONG_DESCRIPTION = """
"""

# code to extract and write the version copied from pandas
FULLVERSION = VERSION
write_version = True

if not ISRELEASED:
    import subprocess
    FULLVERSION += '.dev'

    pipe = None
    for cmd in ['git', 'git.cmd']:
        try:
            pipe = subprocess.Popen(
                [cmd, "describe", "--always", "--match", "v[0-9]*"],
                stdout=subprocess.PIPE)
            (so, serr) = pipe.communicate()
            if pipe.returncode == 0:
                break
        except:
            pass

    if pipe is None or pipe.returncode != 0:
        # no git, or not in git dir
        if path.exists('relic/version.py'):
            warnings.warn("WARNING: Couldn't get git revision, using existing "
                          "phoeton/version.py")
            write_version = False
        else:
            warnings.warn("WARNING: Couldn't get git revision, using generic "
                          "version string")
    else:
        # have git, in git dir, but may have used a shallow clone (travis)
        rev = so.strip()
        # makes distutils blow up on Python 2.7
        if sys.version_info[0] >= 3:
            rev = rev.decode('ascii')

        if not rev.startswith('v') and re.match("[a-zA-Z0-9]{7,9}", rev):
            # partial clone, manually construct version string
            # this is the format before we started using git-describe
            # to get an ordering on dev version strings.
            rev = "v%s.dev-%s" % (VERSION, rev)

        # Strip leading v from tags format "vx.y.z" to get th version string
        FULLVERSION = rev.lstrip('v').replace(VERSION + '-', VERSION + '+')
else:
    FULLVERSION += QUALIFIER


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
isreleased = %s
"""
    if not filename:
        filename = path.join(path.dirname(__file__), 'relic', 'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION, ISRELEASED))
    finally:
        a.close()


if write_version:
    write_version_py()


def file_walk(top, remove=''):
    """
    Returns a generator of files from the top of the tree, removing
    the given prefix from the root/file result.
    """
    top = top.replace('/', path.sep)
    remove = remove.replace('/', path.sep)
    for root, dirs, files in walk(top):
        for file in files:
            yield path.join(root, file).replace(remove, '')


setup(
    # Project info
    name=DISTNAME,
    version=FULLVERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    # The project's main homepage.
    url=URL,
    # Author details
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    # License
    license=LICENSE,
    classifiers=CLASSIFIERS,
    # What does your project relate to?
    keywords=['oggm'],
    # We are a python 3 only shop
    python_requires='>=3.4',
    # Find packages automatically
    packages=find_packages(exclude=['docs']),
    # let pip install the dependencies. brutal but convenient
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'oggm',
        'xarray',
        ],
    # additional groups of dependencies here (e.g. development dependencies).
    extras_require={},
    # data files that need to be installed
    package_data={},
    # Old
    data_files=[],
    # Executable scripts
    entry_points={},
)
