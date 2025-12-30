using System;
using System.Linq;
using System.Text;
using System.Collections.Generic;
using Accord.MachineLearning.DecisionTrees;
using Accord.MachineLearning.DecisionTrees.Learning;
using FraudDetection.ML.Classifiers;

namespace FraudDetection.ML.Classifiers
{
    // random forest classifier implementation         binary classification  using ensemble of decision trees
    public class RandomForestClassifier : Classifier
    {
        
        private RandomForest randomForest;
        
        // number of trees in the forest
        private int   numberOfTrees;
        
        // number of variables to consider at each split
        // if 0, uses the square root of the total number of features
        private int variablesToConsider;
        
        // sampling ratio for each tree (bootstrap sampling)
        // value between 0.0 and 1.0, default is 1.0 (use all samples with replacement)
        private double sampleRatio;
    
        private int randomSeed;
        
        // initializes a new instance of the random forest classifier class
        public RandomForestClassifier( int NumberOfTrees = 100, int VariablesToConsider = 0, 
                                    double SampleRatio = 1.0, int RandomSeed = -1)
            : base("Random Forest")
        {
                
            numberOfTrees= NumberOfTrees;
            variablesToConsider = VariablesToConsider;
            sampleRatio = SampleRatio;
            randomSeed = RandomSeed;
            randomForest = null;
        }
        
        //gets the number of trees 
        public int NumberOfTrees 
        { 
            get { return numberOfTrees; }      }
        
        // gets the number of variables considered at each split
        public int VariablesToConsider 
        { 
            get { return variablesToConsider; }    }
        
        // gets the sampling ratio used for bootstrap 
        public double SampleRatio 
        { 
            get { return sampleRatio; } 
        }
        
        //gets the random seed used for training (-1 if non  deterministic)
        public int RandomSeed 
        { 
            get { return randomSeed; }    }
        
        // trains the random forest classifier on the provided feature data and labels
        public override void Train(double[][] features, int[] labels)
        {
            try
            {
                // Store feature count for future validation
                numberOfFeatures = features[0].Length;
                
                // Create and train the random forest
                RandomForestLearning teacher = new RandomForestLearning();
                teacher.NumberOfTrees = numberOfTrees;
                
                randomForest = teacher.Learn(features, labels);
                isTrained = true;
                
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($" Training failed : {ex.Message}", ex);
            }
        }
        
        // makes a single prediction using the trained random forest model
        protected override int Decide(double[] features)
        {
            if (!isTrained)
                throw new InvalidOperationException("  Model must be trained before making predictions!");
                
                
            try
            {
                //use accord Decide method
                return randomForest.Decide(features);
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Prediction failed: {ex.Message}", ex);
            }
        }
        
        //gets the probability of fraud for a single feature vector
        public override double GetProbability(double[] features)
        {
            if (!isTrained)
                throw new InvalidOperationException("Model must be trained before calculating probabilities");
                
                
            try
            {
                //compute  the   probability by counting tree votes
                if (randomForest.Trees == null || randomForest.Trees.Length == 0)
                    throw new InvalidOperationException("Random Forest has no trained trees");
                
                int positiveVotes = 0;
                int totalTrees = randomForest.Trees.Length;
                
                foreach (DecisionTree tree in randomForest.Trees)
                {
                    if (tree != null)
                    {
                        int treeDecision = tree.Decide(features);
                        if (treeDecision == 1) // vote for positive class
                            positiveVotes++;
                    }
                }
                
                //calculate the  probability as proportion of positive votes
                double probability = (double)positiveVotes / totalTrees;
                
                return probability;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Probability calculation failed: {ex.Message}", ex);
            }
        }
        
        // gets the probabilities of fraud for multiple feature vectors
        // processes all samples efficiently in batch 
        public override double[] GetProbability(double[][] features)
        {
            if (!isTrained)
                throw new InvalidOperationException("Model must be trained before calculating probabilities");
                
                
            try
            {
                //process each sample individually for probability calculation
                double[] probabilities = new double[features.Length];
                
                for (int i = 0; i < features.Length; i  ++)
                {
                    probabilities[i] = GetProbability(features[i]);
                }
                
                return probabilities;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Batch probability calculation failed: {ex.Message}", ex);
            }
        }
        
        // displays  information about the random forest model
        public override void DisplayModelInfo()
        {
            Console.WriteLine("=== Random Forest Model Information ===");
            Console.WriteLine($"Model Name: {modelName}");
            string trainingStatus;
            if (isTrained)
            {
                trainingStatus = "Trained";
            }
            else
            {
                trainingStatus = "Not Trained";
            }
            Console.WriteLine($"Training Status: {trainingStatus}");
            Console.WriteLine($"Decision Threshold: {decisionThreshold:F3}");
            Console.WriteLine($"Number of Trees: {numberOfTrees}");
            string autoText;
            if (variablesToConsider == 0)
            {
                autoText = "(auto)";
            }
            else
            {
                autoText = "";
            }
            Console.WriteLine($"Variables to Consider: {variablesToConsider} {autoText}");
            Console.WriteLine($"Sample Ratio: {sampleRatio:F3}");
            string randomSeedText;
            if (randomSeed == -1)
            {
                randomSeedText = "None (non-deterministic)";
            }
            else
            {
                randomSeedText = randomSeed.ToString();
            }
            Console.WriteLine($"Random Seed: {randomSeedText}");
            
            if (isTrained )
            {
                Console.WriteLine($"Number of Features: {numberOfFeatures}");
                Console.WriteLine($"Number of Classes: {numberOfClasses}");
                
                // display additional forest statistics if available
                if (randomForest != null && randomForest.Trees != null)
                {
                    Console.WriteLine($"Actual Trees in Forest: {randomForest.Trees.Length}");
                    
                    // calculate average tree depth
                    int totalDepth = 0 ;
                    int validTrees = 0;
                    
                    foreach (DecisionTree tree in randomForest.Trees)
                    {
                        if (tree != null)
                        {
                            totalDepth +=   tree.GetHeight();
                            validTrees++;
                        }
                    }
                    
                    if (validTrees > 0)
                    {
                        double averageDepth = (double)totalDepth / validTrees;
                        Console.WriteLine($"Average Tree Depth: {averageDepth:F2}");
                    }
                }
            }
            else
            {
                Console.WriteLine("Model not trained - detailed information not available");
            }
            
            Console.WriteLine("   =====================================  ");
        }
        
        // gets  model information for display purposes
        //summary of the trained random forest model
        public string GetModelInfo()
        {
            if (!IsTrained)
                return "Model not trained";
            
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Random Forest Classifier (Trees: {numberOfTrees})");
            sb.AppendLine($"Model Name: {ModelName}");
            sb.AppendLine($"Decision Threshold: {DecisionThreshold:F3}");
            sb.AppendLine($"Number of Features: {NumberOfFeatures}");
            sb.AppendLine($"Number of Classes: {NumberOfClasses}");
            
            if (randomForest != null && randomForest.Trees != null)
            {
                int[] treeHeights = randomForest.Trees.Select(tree => tree.GetHeight()).ToArray();
                sb.AppendLine($"Number of Trees: {randomForest.Trees.Length}" );
                sb.AppendLine($"Average Tree Height: {treeHeights.Average():F1}");
                sb.AppendLine($"Max Tree Height: {treeHeights.Max()}" );
                sb.AppendLine($"Min Tree Height: {treeHeights.Min()}");
            }
            
            return sb.ToString();
        }
        
    }
} 