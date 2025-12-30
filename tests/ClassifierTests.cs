using System;
using FraudDetection.ML.Classifiers;

namespace FraudDetection.Tests
{
    // tests logistic regression and random forest classifiers
    public class ClassifierTests
    {
        // runs tests for both classifiers
        public static void RunAllTests()
        {
            Console.WriteLine("=== machine Learning Classifier Tests ===");
            Console.WriteLine();
            
            // test logistic regression
                TestLogisticRegression();
                Console.WriteLine();
                
            // test random forest
                TestRandomForest();
                Console.WriteLine();
                
            Console.WriteLine("all tests completed");
        }
        
        // tests the logistic regression classifier
        private static void TestLogisticRegression()
        {
            Console.WriteLine("testing logistic regression classifier...");
            
            // create sample data
            double[][] features = CreateSampleData();
            int[] labels = CreateSampleLabels();
            
            // create classifier
            LogisticRegressionClassifier classifier = new LogisticRegressionClassifier();
            
            // train the model
            Console.WriteLine("training model...");
            classifier.Train(features, labels);
            
            // display results
            Console.WriteLine("training completed");
            Console.WriteLine($" model name: {classifier.ModelName}");
            Console.WriteLine($"number of features: {classifier.NumberOfFeatures}");
            Console.WriteLine($"is trained: {classifier.IsTrained}");
            
            // calculate accuracy
            double accuracy = classifier.GetAccuracy(features, labels);
            Console.WriteLine($"accuracy: {accuracy:P2}");
            
            // display model info
            classifier.DisplayModelInfo();
            
            Console.WriteLine("logistic regression test completed");
        }
        
        // tests the random forest classifier
        private static void TestRandomForest()
        {
            Console.WriteLine("testing random forest classifier...");
            
            // create sample data
            double[][] features = CreateSampleData();
            int[] labels = CreateSampleLabels();
            
            // create classifier
            RandomForestClassifier classifier = new RandomForestClassifier(NumberOfTrees: 10, RandomSeed: 42);
            
            // train      model
            Console.WriteLine("training model...");
            classifier.Train(features, labels);
            
            // display results
            Console.WriteLine("training completed");
            Console.WriteLine($"model name: {classifier.ModelName}");
            Console.WriteLine($"number of trees: {classifier.NumberOfTrees}");
            Console.WriteLine($"number of features: {classifier.NumberOfFeatures}");
            Console.WriteLine($"is trained: {classifier.IsTrained}");
            
            // calculate accuracy
            double accuracy = classifier.GetAccuracy(features, labels);
            Console.WriteLine($"accuracy: {accuracy:P2}");
            
            // display model info
            classifier.DisplayModelInfo();
            
            Console.WriteLine("random forest test completed");
        }
        
        // creates some sample data for testing
        private static double[][] CreateSampleData()
        {
            return new double[][]
            {
                // not fraud (class 0)
                new double[] { 0.1, 0.2, 0.0, 1.0 },
                new double[] { 0.2, 0.1, 0.0, 1.0 },
                new double[] { 0.15, 0.25, 0.0, 1.0 },
                new double[] { 0.05, 0.15, 0.0, 1.0 },
                new double[] { 0.3, 0.2, 0.0, 1.0 },
                
                // fraud 
                new double[] { 0.8, 0.9, 1.0, 0.0 },
                new double[] { 0.9, 0.8, 1.0, 0.0 },
                new double[] { 0.85, 0.85, 1.0, 0.0 },
                new double[] { 0.75, 0.95, 1.0, 0.0 },
                new double[] { 0.95, 0.75, 1.0, 0.0 }
            };
        }
        
        // creates sample labels for the data
        private static int[] CreateSampleLabels()
        {
            return new int[]
            {
                // not fraud (class 0)
                0, 0, 0, 0, 0,
                
                // fraud (class 1)
                1, 1, 1, 1, 1
            };
        }
    }
} 