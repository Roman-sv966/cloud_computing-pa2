## Cloud Computing PA2

### Prerequisites
1. Access to your AWS instance
2. Required files:
   - Application JAR file
   - Training dataset
   - Validation dataset

### Setup
1. Log in to your AWS Management Console

2. Upload the following to your S3 bucket:
   - JAR file
   - Training data
   - Validation data

3. Create an EMR cluster with spark enabled

### Model Training
4. To train the model:
   - Navigate to the EMR cluster dashboard
   - Select "Steps" and click "Add step"
   - Choose "Spark Application"
   - Configure the step:
     - Provide a meaningful name
     - Select your JAR file
     - Add `--class com.example.Train` to spark-submit options
   - Click "Add step" to begin training
   
The training process will create and save the best performing model to S3.

### Model Testing
5. To test the trained model:
   - Navigate to the EMR cluster dashboard
   - Select "Steps" and click "Add step"
   - Choose "Spark Application"
   - Configure the step:
     - Provide a meaningful name
     - Select your JAR file
     - Add `--class com.example.Test` to spark-submit options
   - Click "Add step" to begin testing

### Docker Deployment

#### Building the Image
6. Authenticate with Docker Hub:
```bash
docker login
```

7. Build the Docker image:
```bash
docker build -t <docker-hub-username>/cloud_computing-pa2 .
```

8. Push to Docker Hub:
```bash
docker push <docker-hub-username>/cloud_computing-pa2
```

#### Local Testing
9. Prepare your test environment:
   - Download the trained model to `top_model/`
   - Place your test data in `ValidationDataset.csv`

10. Run inference locally:
```bash
docker run -v $(pwd):/data <docker-hub-username>/cloud_computing-pa2 spark-submit --class com.example.Test /data/cloud_computing-pa2-1.0.jar /data/best_model /data/ValidationDataset.csv
```
