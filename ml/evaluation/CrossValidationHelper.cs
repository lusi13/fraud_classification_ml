using System;
using System.Collections.Generic;
using System.Linq;
using FraudDetection.ML.Interfaces;
using FraudDetection.ML.Classifiers;

namespace FraudDetection.ML.Evaluation
{
    // kfold to split data into k parts, use k-1 for training, 1 for testing, repeat k times
    // just returns average test accuracy
    public class CrossValidationHelper
    {
        private int folds;
        
        public CrossValidationHelper(int numberOfFolds = 3)
        {
            if (numberOfFolds <2)
                throw new ArgumentException("need at least 2 folds!!");
            
            folds = numberOfFolds;
        }
        
        public int GetNumberOfFolds()
        {
            return folds;  }
        
        // evaluate and returns average test accuracy
        public double Evaluate(IClassifier classifier, double[][] features, int[] labels, bool quiet = false)
        {
            if (!quiet)
                Console.WriteLine($"doing the {folds}-fold cross validation for { classifier.ModelName}");
            
            List<double> testAccuracies= new List<double>();
            List<double> testRecalls = new List<double>();
            int foldSize = features.Length / folds;
            int successfulFolds = 0;
            
            for (int fold= 0; fold < folds; fold ++ )
            {
                try
                {
                    //figure out which samples to use for testing this fold
                    int testStart = fold * foldSize;
                    int testEnd;
                    if (fold == folds - 1)
                    {
                        testEnd = features.Length;
                    }
                    else
                    {
                        testEnd = (fold + 1) * foldSize;
                    }
                    
                    // split data into train and test lists
                    List<double[]> trainFeatures = new List<double[]>();
                    List<int> trainLabels = new List<int>();
                    List<double[]> testFeatures = new List<double[]>();
                    List<int> testLabels = new List<int>();
                    
                    for (int i = 0; i < features.Length; i++)
                    {
                        if (i >= testStart && i < testEnd)
                        {
                            // this sample goes to test set
                            testFeatures.Add(features[i]);
                            testLabels.Add(labels[i]);
                        }
                        else
                        {
                            // this sample goes to training set
                            trainFeatures.Add(features[i]);
                            trainLabels.Add(labels[i]);
                        }
                    }
                    
                    // convert lists to arrays
                    double[][] trainFeaturesArray = trainFeatures.ToArray();
                    int[] trainLabelsArray= trainLabels.ToArray();
                    double[][] testFeaturesArray = testFeatures.ToArray();
                    int[] testLabelsArray = testLabels.ToArray();
                    
                    //for Random Forest, remove constant features to prevent training errors
                    if (classifier.ModelName.Contains("Random Forest"))
                    {
                        var (filteredTrainFeatures, filteredTestFeatures) = RemoveConstantFeatures(trainFeaturesArray, testFeaturesArray);
                        trainFeaturesArray = filteredTrainFeatures;
                        testFeaturesArray = filteredTestFeatures;
                    }
                    
                    //check if we have both classes in training data
                    bool hasClass0 = trainLabelsArray.Any(label => label == 0);
                    bool hasClass1 = trainLabelsArray.Any(label => label == 1);
                    
                    if (!hasClass0 || !hasClass1)
                    {
                        if (!quiet)
                            Console.WriteLine($"fold {fold + 1}: SKIPPED (missing class in training data)");
                        continue;
                    }
                    
                    // train classifier on training data
                    classifier.Train(trainFeaturesArray, trainLabelsArray);
                    
                    // test on testing data
                    double accuracy = classifier.GetAccuracy(testFeaturesArray, testLabelsArray);
                    double recall = CalculateRecall(classifier, testFeaturesArray, testLabelsArray);
                    testAccuracies.Add(accuracy);
                    testRecalls.Add(recall);
                    successfulFolds++;
                    
                    if (!quiet)
                        Console.WriteLine($"fold {fold + 1}: accuracy={accuracy:P2}, recall={recall:P2}");
                }
                catch (Exception ex)
                {
                    if (!quiet)
                        Console.WriteLine($"fold {fold + 1}: SKIPPED ({ex.Message})");
                    // Continue to next fold
                    continue;
                }
            }
            
            if (testAccuracies.Count == 0)
            {
                if (!quiet)
                    Console.WriteLine("ERROR: All folds failed!!!!!!");
                return 0.0;
            }
            
            double avgAccuracy = testAccuracies.Average();
            double avgRecall = testRecalls.Average();
            if (!quiet)
            {
                Console.WriteLine($"average accuracy: {avgAccuracy:P2}, average recall: {avgRecall:P2} (from {successfulFolds}/{folds} successful folds)");
                Console.WriteLine();
            }
            
            return avgRecall ; // return recall for model selection
        }
        
        // calculate recall = true positive rate
        private double CalculateRecall(IClassifier classifier, double[][] testFeatures, int[] testLabels)
        {
            int[] predictions = classifier.Predict(testFeatures);
            
            int truePositives = 0;
            int falseNegatives = 0;
            
            for (int i = 0; i < testLabels.Length; i++)
            {
                if (testLabels[i] == 1 && predictions[i] == 1)
                    truePositives++;
                else if (testLabels[i] == 1 && predictions[i] == 0)
                    falseNegatives++;
            }
            
            int totalFraudCases= truePositives + falseNegatives;
            if (totalFraudCases == 0)
                return 0.0; // no fraud cases in test set
                
            return (double)truePositives / totalFraudCases;
        }
        
        // simple method to remove features that have the same value for all samples
        // this prevents attribute is a constant errors in random forest
        private (double[][], double[][]) RemoveConstantFeatures(double[][] trainFeatures, double[][] testFeatures)
        {
            if (trainFeatures.Length == 0) return (trainFeatures, testFeatures);
            
            List<int> nonConstantIndices = new List<int>();
            
            // check each feature to see if its constant
            for (int featureIndex = 0; featureIndex < trainFeatures[0].Length; featureIndex++  )
            {
                double firstValue = trainFeatures[0][featureIndex];
                bool isConstant = true;
                
                // check if all training samples have the same value for this feature
                for (int sampleIndex = 1; sampleIndex < trainFeatures.Length; sampleIndex++)
                {
                    if (trainFeatures[sampleIndex][featureIndex] != firstValue)
                    {
                        isConstant= false;
                        break;
                    }
                }
                
                // if feature is not constant, keep it
                if (!isConstant )
                {
                    nonConstantIndices.Add( featureIndex);
                }
            }
            
            // if all features are non-constant, return original data
            if (nonConstantIndices.Count == trainFeatures[0].Length)
            {
                return (trainFeatures, testFeatures);    }
            
            // create new arrays with only non-constant features
            double[][] newTrainFeatures = new double[trainFeatures.Length][];
            double[][] newTestFeatures = new double[testFeatures.Length][];
            
            for (int i = 0; i < trainFeatures.Length; i++)
            {
                newTrainFeatures[i] = new double[nonConstantIndices.Count];
                for (int j = 0 ; j < nonConstantIndices.Count; j++ )
                {
                    newTrainFeatures[i][j] = trainFeatures[i][nonConstantIndices[j]];
                }
            }
            
            for (int i = 0 ; i < testFeatures.Length; i++ )
            {
                newTestFeatures[i] = new double[nonConstantIndices.Count];
                for (int j = 0; j < nonConstantIndices.Count; j++)
                {
                    newTestFeatures[i][j] = testFeatures[i][nonConstantIndices[j]];
                }
            }
            
            return (newTrainFeatures, newTestFeatures );
        }

    }
} 