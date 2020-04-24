# Deployment Types
<img src="https://i.imgur.com/kg6Nyxw.png" height="300" />

# Create Endpint
<img src="https://i.imgur.com/cDRjUwk.png" height="300" />

# Steps for Deploying a Model

__One__ Create model
1. S3 for model artifact
2. Docker registry Path
3. name that can be used for subsequent deployment stages

__Two__ Create Endpoint config
1. name
2. ML compute instance for hosting model

__Three__ Create Endpoint 

## How to serve two result from 2 models
- API gatway -> Lambda -> fetch multiple results

## Where to check error if Training job if implemnted by pipeline
- DescribeJob API
- CloudWatch
- Console DOEST NOT HELP here