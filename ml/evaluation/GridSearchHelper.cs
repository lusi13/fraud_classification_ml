using System;
using System.Collections.Generic;
using FraudDetection.ML.Classifiers;

namespace FraudDetection.ML.Evaluation
{
    public static class GridSearchHelper
    {
        public static LogisticRegressionClassifier FindBestLogisticRegressionClassifier(
            double[][] features, int[] labels, int numberOfFolds = 3 )
        {
            // test bigger range of values
            double[] tolerances = { 1e-8, 1e-6, 1e-4};
            int[] maxIterations = {100, 200 , 300};
            
            double bestRecall = 0.0;
            double bestAccuracy= 0.0;
            LogisticRegressionClassifier bestClassifier =  null;
            
            Console.WriteLine(" === Logistic Regression Grid Search ===");
            Console.WriteLine($"using {numberOfFolds}-fold cross-validation");
            Console.WriteLine($" testing {tolerances.Length * maxIterations.Length} parameter combinations..");
            Console.WriteLine();
            
            for (int i = 0; i < tolerances.Length; i ++ )
            {
                for (int j = 0; j < maxIterations.Length ; j ++)
                {
                    Console.Write($"testing tolerance={tolerances[i]:E2}, max_iterations ={ maxIterations[j]} ..... ");
                    
                    LogisticRegressionClassifier classifier = new LogisticRegressionClassifier( tolerances[i ], maxIterations[j]);
                     
                    CrossValidationHelper cv = new CrossValidationHelper( numberOfFolds);
                    double recall = cv.Evaluate(classifier, features, labels, quiet: true);
                    
                    // calculate accuracy separately for display
                    classifier.Train(features, labels);
                    double accuracy = classifier.GetAccuracy(features, labels);
                    
                    Console.WriteLine($"accuracy: {accuracy:P2}, recall: {recall:P2}");
                    
                    if (recall > bestRecall)
                    {
                        bestRecall = recall;
                        bestAccuracy = accuracy;
                        bestClassifier = new LogisticRegressionClassifier(tolerances[i] , maxIterations[j]);
                    }
                }
            }
            
            Console.WriteLine();
            if (bestClassifier != null)
            {
                bestClassifier.Train(features, labels);
                Console.WriteLine("best logistic regression found:");
                Console.WriteLine($"  tolerance: {bestClassifier.ConvergenceTolerance:E2}");
                Console.WriteLine($"  max iterations: {bestClassifier.MaxIterations}");
                Console.WriteLine($"  cross validation accuracy: {bestAccuracy:P2}");
                Console.WriteLine($"  cross-validation recall: {bestRecall:P2}");
            }
            else
            {
                Console.WriteLine(" no valid logistic regression classifier found");
            }
            Console.WriteLine();
            return bestClassifier;
        }
        
        public static RandomForestClassifier FindBestRandomForestClassifier(
            double[][] features, int[] labels ,  int numberOfFolds = 3)
        {
            // test broader range of values
            int[] numberOfTrees = {50, 100, 200};
            double[] sampleRatios = {0.7,  0.9, 1.0 };
            
            double bestRecall = 0.0;
            double bestAccuracy = 0.0;
            RandomForestClassifier bestClassifier = null;
            
            Console.WriteLine("=== Random Forest Grid Search ===");
            Console.WriteLine($"using {numberOfFolds}-fold cross validation");
            Console.WriteLine($"testing {numberOfTrees.Length * sampleRatios.Length} parameter combinations..");
            Console.WriteLine();
            
            for (int i = 0; i < numberOfTrees.Length ;  i++)
            {
                for (int j = 0; j < sampleRatios.Length; j++ )
                {
                    Console.Write($"testing trees={ numberOfTrees[i]}, sample_ratio =  {sampleRatios[j]:F1}..... " );
                    
                    RandomForestClassifier classifier = new RandomForestClassifier(numberOfTrees[i], 0, sampleRatios[j], -1);
                    
                    CrossValidationHelper cv = new CrossValidationHelper(numberOfFolds);
                    double recall = cv.Evaluate(classifier, features, labels, quiet: true);
                    
                    // calculate accuracy separately for display
                    classifier.Train(features, labels);
                    double accuracy = classifier.GetAccuracy( features, labels);
                    
                    Console.WriteLine($"accuracy: { accuracy:P2}, recall: {recall:P2}");
                    
                    if (recall > bestRecall)
                    {
                        bestRecall = recall;
                        bestAccuracy = accuracy;
                        bestClassifier = new RandomForestClassifier(numberOfTrees[i], 0, sampleRatios[j], -1);
                    }
                }
            }
            
            Console.WriteLine();
            if (bestClassifier != null)
            {
                bestClassifier.Train(features, labels);
                Console.WriteLine("best random forest found:");
                Console.WriteLine($"  number of trees: {bestClassifier.NumberOfTrees}");
                Console.WriteLine($"  sample ratio: {bestClassifier.SampleRatio:F1}" );
                Console.WriteLine($"  cross-validation accuracy: {bestAccuracy:P2}"  );
                Console.WriteLine($"  cross  validation recall: {bestRecall:P2}");
            }
            else
            {
                Console.WriteLine("no valid random forest  classifier found !");
            }
            Console.WriteLine();
            return bestClassifier;
        }
    }
} 