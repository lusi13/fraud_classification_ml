using System;
using FraudDetection.Tests;
using FraudDetection.Data;
using FraudDetection.Entities;
using FraudDetection.ML.Classifiers;
using FraudDetection.ML.Evaluation;
using System.Collections.Generic;
using System.Linq;

class Program
{
    static void Main(string[] args)
    {
        // uncomment to run the  tests
        //Console.WriteLine("running classifiers test......");
        //ClassifierTests.RunAllTests();
        //return;
        //Console.WriteLine("running dataloading test......");
        //DataLoaderTests.RunTests();
        //return;

        Console.WriteLine("=== Fraud Detection System ===");
        Console.WriteLine();
        
        try
        {
            // load data
            string datasetPath = "dataset/dataset.csv";
            List<InsuranceRecord> records = DataLoader.LoadDataset(datasetPath);
            ProcessedDataset processedData = DataPreprocessor.PreprocessData(records);
            
            (double[][] trainFeatures, int[] trainLabels, double[][] testFeatures, int[] testLabels) = 
                DataPreprocessor.SplitTrainTest(processedData.Features, processedData.Labels, testRatio: 0.2);
            
            Console.WriteLine($"dataset loaded: {trainFeatures.Length} training samples, {testFeatures.Length} test samples");
            
            // count label distribution
            int trainFraud = 0;
            int trainNonFraud = 0;
            int testFraud = 0;
            int testNonFraud = 0;
            
            // count training labels
            for (int i = 0; i < trainLabels.Length; i++)
            {
                if (trainLabels[i] == 0)
                    trainNonFraud++;
                else
                    trainFraud  ++;
            }
            
            // count test labels
            for (int i = 0; i < testLabels.Length; i++)
            {
                if (testLabels[i] == 0)
                    testNonFraud++;
                else
                    testFraud++;
            }
            
            Console.WriteLine();
            Console.WriteLine("=== label distribution   ===");
            Console.WriteLine($"training set ({trainLabels.Length} samples):");
            Console.WriteLine($"  non fraud (0): {trainNonFraud} ({(double)trainNonFraud/trainLabels.Length:P1})");
            Console.WriteLine($"  fraud (1):     {trainFraud} ({(double)trainFraud/trainLabels.Length:P1})");
            Console.WriteLine($"test set ({testLabels.Length} samples):");
            Console.WriteLine($"  non fraud (0): {testNonFraud} ({(double)testNonFraud/testLabels.Length:P1})");
            Console.WriteLine($"  fraud (1):     {testFraud} ({(double)testFraud/testLabels.Length:P1})");
            Console.WriteLine();
            
            // logistic regression grid search
            LogisticRegressionClassifier bestLR = GridSearchHelper.FindBestLogisticRegressionClassifier(trainFeatures, trainLabels);
            bestLR.DisplayModelInfo();

            Console.WriteLine();
            
            // random forest grid search
            RandomForestClassifier bestRF = GridSearchHelper.FindBestRandomForestClassifier(trainFeatures, trainLabels);
            bestRF.DisplayModelInfo();

            Console.WriteLine();
            Console.WriteLine("===   final test results ===    ");
            
            double lrTestAccuracy = bestLR.GetAccuracy(testFeatures, testLabels);
            Console.WriteLine($"logistic regression test accuracy: {lrTestAccuracy:P2}");
            
            double rfTestAccuracy = bestRF.GetAccuracy(testFeatures, testLabels);
            Console.WriteLine($"random forest test accuracy: {rfTestAccuracy:P2}");
            
            // calculate and display detailed metrics for both models
            Console.WriteLine();
            Console.WriteLine("===      detailed metrics ===");
            
            int[] lrPredictions = bestLR.Predict(testFeatures);
            int[] rfPredictions = bestRF.Predict(testFeatures);
            
            PrintSimpleMetrics("logistic regression", testLabels, lrPredictions);
            Console.WriteLine();
            PrintSimpleMetrics("  random forest", testLabels, rfPredictions  );
            Console.WriteLine();
        }
        catch (Exception ex)
        {
            Console.WriteLine($"grid search test failed: {ex.Message }");
        }
    }
    
    private static void PrintSimpleMetrics(string modelName, int[] actualLabels, int[] predictions)
    {
        // calculate confusion matrix
        int truePositives = 0;
        int trueNegatives = 0;
        int falsePositives = 0;
        int falseNegatives = 0;
        
        for (int i = 0; i < actualLabels.Length; i  ++)
        {
            if (actualLabels[i] == 1 && predictions[i] == 1) 
                truePositives++;
            else if (actualLabels[i] == 0 && predictions[i] == 0) 
                trueNegatives++;
            else if (actualLabels[i] == 0 && predictions[i] == 1) 
                falsePositives++;
            else if (actualLabels[i] == 1 && predictions[i] == 0) 
                falseNegatives++;
        }
        
        // calculate core metrics
        double accuracy = (double)(truePositives + trueNegatives) / actualLabels.Length;
        
        double precision;
        if (truePositives + falsePositives > 0)
            precision = (double)truePositives / (truePositives + falsePositives);
        else
            precision = 0.0;
            
        double recall;
        if (truePositives + falseNegatives > 0)
            recall = (double)truePositives / ( truePositives + falseNegatives);
        else
            recall = 0.0;
            
        double f1Score;
        if ( precision + recall >  0   )  
            f1Score = 2 * (precision * recall) / (precision + recall );
        else
            f1Score = 0.0;
        
        // display results
        Console.WriteLine($"{modelName} Results:");
        Console.WriteLine($"  Accuracy:  {accuracy:P2}");
        Console.WriteLine($"  Precision: {precision:P2}");
        Console.WriteLine($"  Recall:    {recall:P2}");
        Console.WriteLine($"  F1 Score:  {f1Score:P2}");
        Console.WriteLine($"  Confusion Matrix: TP={truePositives}, TN={trueNegatives}, FP={falsePositives}, FN={falseNegatives}");
    }
}

