# Create a base image with all of our requirements installed.
FROM python:3.8.5 as base
COPY ./requirements.txt /librl/install/requirements.txt
RUN pip install -r /librl/install/requirements.txt

# Copy over source files, build python package.
FROM base as build
COPY . /librl/source
WORKDIR /librl/source
RUN pip install -e .
