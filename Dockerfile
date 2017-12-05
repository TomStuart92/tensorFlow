FROM ubuntu

# Install updates
RUN apt-get update

# Install curl
RUN apt-get install python python-pip python-dev python-numpy -y

# Install tensorflow
RUN pip install tensorflow

# Copy server
COPY . /app
WORKDIR /app
