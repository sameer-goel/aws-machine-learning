# Train and Deployment using SageMaker

<img src="https://i.imgur.com/OaE8uHm.png" width="500" />

# Deployment Types
<img src="https://i.imgur.com/kg6Nyxw.png" height="300" />

# Create Endpoint
<img src="https://i.imgur.com/cDRjUwk.png" height="300" />

# Steps for Deploying a Model

__1__ Create model
1. S3 for model artifact
2. Docker registry Path
3. name that can be used for subsequent deployment stages

__2__ Create Endpoint config
1. name
2. ML compute instance for hosting model

__3__ Create Endpoint 

## How to serve two result from 2 models
- API gateway -> Lambda -> fetch multiple results

## Where to check error if Training job if implemented by pipeline
- DescribeJob API
- CloudWatch
- Console DOEST NOT HELP here

## Sagemaker tools
<img src="https://d1.awsstatic.com/re19/Sagemaker/SageMaker_Overview-Chart.247eaea6e41ddca8299c5a9a9e91b5d78b751c38.png" height="300" />

## Production Variant

1. A/B Testing with Rollout: Less Risky - Test offline and then deploy as rollout
2. Canary: More Risky - Deploy 10:90 split of real data and shift based on performance.

## Sagemaker Docker containers

After a Docker image is built, it can be pushed to the Amazon Elastic Container Registry (ECR).

<img src="https://i.imgur.com/UbgqDyJ.png" width="500" />
<img src="https://i.imgur.com/MiJ4RiX.png" width="500" />

Models in SageMaker are hosted in Docker containers
- Pre built Models in scikit Learn and Spark ML
- Pre built Models in Tensorflow , MXNet , and PyTorch
- Custom algorithm

### Docker Image Structure
1. nginx.conf : configuration file for nginx front end.
2. wsgi.py A wrapper used to invoke Flask application.
3. predictor.py: program that implements Flask web server
4. Serve: program that run when container is started for hosting.
5. Train: program that is invoked when the container is run
during training.

## Sagemaker Security
By-Default notebooks are internet enabled.

1. S3
- Resource based: ACL
- User-based: IAM, MFA, TAGs
- Macie to find sensitive data in S3

2. Auditing - CloudTrail
3. Logs - CloudWatch
4. VPC Endpoints, connection over private links


### Encryption 
- Transit: TLS/SSL, Inter-container traffic encryption
- Rest: KMS 

## [Update endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/endpoint-scaling.html) that use automatic scaling

1. De-register endpoint as (DeregisterScalableTarget)
2. As auto-scaling is off, as a precaution you can update instance count for production, just in case there us heavy traffic (UpdateEndpointWeightsAndCapacities)
3. Wait for Endpoint to be __InService__ (DescribeEndpoint).
4. Call DescribeEndpointConfig to get the current config.

`Actual Update Steps`
5. Create NEW Endpoint config (__CreateEndpointConfig__)
6. Update endpoint (__UpdateEndpoint__) with latest modelS3 path.
7. Re-enable automatic scaling by(__RegisterScalableTarget__)

## 
