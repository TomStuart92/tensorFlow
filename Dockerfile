FROM ubuntu

# Install updates
RUN apt-get update

# Install curl
RUN apt-get install python python-pip python-dev python-numpy -y
RUN pip install mock flask

# Install tensorflow
RUN pip install tensorflow

# Copy server
COPY ./scripts /app
WORKDIR /app
