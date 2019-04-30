import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CapsComplete",
    version="0.0.1",
    author="Vinay Ramesh",
    author_email="vrr2112@columbia.edu",
    description="package for capsule occclusion completion",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vinoo999/CapsComplete",
    packages=['CapsComplete', 'CapsComplete.data',
              'CapsComplete.models', 'CapsComplete.analysis'],
    classifiers=[
        "Programming Language :: Python :: 3"
    ],
)
