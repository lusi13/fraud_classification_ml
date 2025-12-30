using System;
using FraudDetection.ML.Interfaces;

namespace FraudDetection.ML.Classifiers
{
    // base class that has common methods, ..., that  all of the  ml models need
    public abstract class Classifier : IClassifier
    {
        
        protected string modelName;
        protected bool isTrained;
        
        // the threshold for deciding fraud vs not fraud
        protected double decisionThreshold;
        
        protected int numberOfFeatures;
        protected int numberOfClasses;
        
        protected Classifier( string ModelName)
        {
            if (string.IsNullOrEmpty(ModelName))
                throw new ArgumentNullException(nameof(ModelName), "  model name cannot be null");
                
            modelName = ModelName;
            isTrained = false;
            decisionThreshold = 0.5; // default: 50% probability = fraud
            numberOfFeatures = 0;
            numberOfClasses = 2;
        }
        
        public string ModelName 
        { 
            get { return modelName; } 
        }
        
        public bool IsTrained 
        { 
            get { return isTrained; } 
        }
        
        public double DecisionThreshold 
        { 
            get { return decisionThreshold; }
            set 
            {
                if (value < 0.0 || value > 1.0 )
                    throw new ArgumentOutOfRangeException(nameof(value), "  decision threshold must be between 0.0 and 1.0 ");
                decisionThreshold = value;
            }
        }
        
        public int NumberOfFeatures 
        { 
            get { return numberOfFeatures; }     }
        
        public int NumberOfClasses 
        { 
            get { return numberOfClasses; } 
        }
        
        // each specific classifier implements this differently
        public abstract void Train(double[][] features, int[] labels);
        
        protected abstract int Decide(double[] features);
        public abstract double GetProbability(double[] features);
        public abstract double[] GetProbability(double[][] features);
        
        public virtual int Predict(double[] features)
        {
            if (!isTrained)
                throw new InvalidOperationException(" the model must be trained before making predictions!");
                
            double probability = GetProbability(features );
            if (probability >= decisionThreshold )
            {
                return 1;
            }
            else
            {
                return 0;
            }
        }
        
        public virtual int[] Predict(double[][] features)
        {
            if (!isTrained)
                throw new InvalidOperationException(" the model must be trained before making predictions");
                
            double[] probabilities = GetProbability(features );
            int[] predictions = new int[probabilities.Length];
            
            for (int i = 0; i < probabilities.Length; i ++)
            {
                if (probabilities[i] >= decisionThreshold)
                {
                    predictions[i] = 1;
                }
                else
                {
                    predictions[i] = 0;
                }
            }
            
            return predictions ;
        }
        
        public virtual double GetAccuracy( double[][] testFeatures, int[] testLabels)
        {
            if ( !isTrained)
                throw new InvalidOperationException("  Model must be trained before calculating accuracy");
                
            int[] predictions = Predict(testFeatures);
            int correctPredictions = 0;
            
            for (int i = 0; i < predictions.Length; i++)
            {
                if (predictions[i] == testLabels[i])
                    correctPredictions++;
            }
            
            return (double)correctPredictions / testFeatures.Length;
        }
        
        //displays information about the model like  its name, training status and configuration
        public virtual void DisplayModelInfo()
        {
            Console.WriteLine("=== Model Information ===");
            Console.WriteLine($"Model Name: {modelName}");
            string trainingStatus;
            if (isTrained )
            {
                trainingStatus = "Trained";
            }
            else
            {
                trainingStatus = "Not Trained";
            }
            Console.WriteLine($"Training Status: { trainingStatus}");
            Console.WriteLine($"Decision Threshold: {decisionThreshold:F3}") ;
            
            if (isTrained)
            {
                Console.WriteLine($"Number of Features: { numberOfFeatures}");
                Console.WriteLine($"Number of Classes: {numberOfClasses}");
            }
            else
            {
                Console.WriteLine("Model not trained - feature information not available  ");
            }
            
            Console.WriteLine("========================");
        }
        
        
    }
} 