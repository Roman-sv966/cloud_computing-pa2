package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.GBTClassifier;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.ml.tuning.CrossValidator;
import org.apache.spark.ml.tuning.CrossValidatorModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.classification.RandomForestClassifier;

import java.io.IOException;
import java.util.Arrays;

public class Train {
    // Constants for data paths and model storage
    private static final String TRAINING_DATA_PATH = "s3://cloud-computing-pa2/TrainingDataset.csv";
    private static final String VALIDATION_DATA_PATH = "s3://cloud-computing-pa2/ValidationDataset.csv";
    private static final String MODEL_SAVE_PATH = "s3://cloud-computing-pa2/best_model";

    public static void main(String[] args) throws IOException {
        // Initialize Spark with descriptive application name
        SparkSession spark = SparkSession.builder()
                .appName("Wine Quality Model Training")
                .getOrCreate();

        try {
            // Load and preprocess datasets
            System.out.println("Loading and preprocessing datasets...");
            Dataset<Row> trainData = loadAndProcessData(spark, TRAINING_DATA_PATH);
            Dataset<Row> validationData = loadAndProcessData(spark, VALIDATION_DATA_PATH);

            // Train and evaluate logistic regression model
            evaluateLogisticRegression(trainData, validationData);

            // Train and evaluate random forest model with cross validation
            trainAndSaveRandomForest(trainData);
        } finally {
            spark.stop();
        }
    }

    private static void evaluateLogisticRegression(Dataset<Row> trainData, Dataset<Row> validationData) {
        System.out.println("\nTraining Logistic Regression Model...");
        // Configure and train logistic regression
        LogisticRegression lr = new LogisticRegression()
                .setLabelCol("quality")
                .setFeaturesCol("scaledFeatures")
                .setMaxIter(10)
                .setRegParam(0.3);
        
        LogisticRegressionModel lrModel = lr.fit(trainData);

        // Evaluate model performance
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction");

        double trainAccuracy = evaluator.evaluate(lrModel.transform(trainData));
        double validationAccuracy = evaluator.evaluate(lrModel.transform(validationData));

        System.out.println("Logistic Regression Results:");
        System.out.println("Training Accuracy: " + trainAccuracy);
        System.out.println("Validation Accuracy: " + validationAccuracy);
    }

    private static void trainAndSaveRandomForest(Dataset<Row> trainData) {
        System.out.println("\nTraining Random Forest Model with Cross Validation...");
        // Configure random forest classifier
        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("quality")
                .setFeaturesCol("scaledFeatures");

        // Setup hyperparameter grid for cross validation
        ParamMap[] paramGrid = new ParamGridBuilder()
                .addGrid(rf.numTrees(), new int[]{10, 20, 50})
                .addGrid(rf.maxDepth(), new int[]{4, 8, 16})
                .build();

        // Configure cross validation
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction");

        CrossValidator cv = new CrossValidator()
                .setEstimator(rf)
                .setEvaluator(evaluator)
                .setEstimatorParamMaps(paramGrid)
                .setNumFolds(3);

        // Perform cross validation and get best model
        System.out.println("Performing cross validation...");
        CrossValidatorModel cvModel = cv.fit(trainData);
        double bestScore = cvModel.avgMetrics()[0];
        System.out.println("Best Model F1 Score: " + bestScore);

        // Save the best model
        try {
            RandomForestClassificationModel bestModel = (RandomForestClassificationModel) cvModel.bestModel();
            bestModel.write().overwrite().save(MODEL_SAVE_PATH);
            System.out.println("Best model saved successfully to: " + MODEL_SAVE_PATH);
        } catch (Exception e) {
            System.err.println("Error saving model: " + e.getMessage());
        }
    }

    private static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        // Feature columns for the wine quality dataset
        String[] columns = {
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
            "pH", "sulphates", "alcohol", "quality"
        };

        // Load and preprocess dataset
        Dataset<Row> df = spark.read()
                .option("header", "true")
                .option("sep", ";")
                .option("inferSchema", "true")
                .csv(filePath)
                .toDF(columns);

        return createAndApplyPipeline(df, columns);
    }

    private static Dataset<Row> createAndApplyPipeline(Dataset<Row> df, String[] columns) {
        // Configure feature assembly and scaling
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");

        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);

        // Create and execute pipeline
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{assembler, scaler});
        
        return pipeline.fit(df).transform(df);
    }
}
