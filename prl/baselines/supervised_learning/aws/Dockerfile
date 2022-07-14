# Pull the base image with python 3.8 as a runtime for your Lambda
FROM public.ecr.aws/lambda/python:3.8

# Install the baselines package from git so that the model can be loaded
# # 1. Install git
RUN yum install git -y

# 2. clone prl_baselines repo
RUN git clone -b aws_lambda https://github.com/hellovertex/prl_baselines.git tmp/prl_baselines
# 3. install the prl_baselines python package locally
RUN cd tmp/prl_baselines && pip install .

# Copy the earlier created requirements.txt file to the container
# COPY requirements.txt ./

# Install the python requirements from requirements.txt
# RUN python3.8 -m pip install -r requirements.txt

# Copy the earlier created app.py file to the container
COPY app.py ./

# Load the BERT model from Huggingface and store it in the model directory
COPY model.pt ./tmp/



# Set the CMD to your handler
CMD ["app.lambda_handler"]