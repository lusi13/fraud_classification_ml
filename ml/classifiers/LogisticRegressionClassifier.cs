using System;
using Accord.Statistics.Models.Regression;
using Accord.Statistics.Models.Regression.Fitting;
using FraudDetection.ML.Classifiers;

namespace FraudDetection.ML.Classifiers
{
  
    public class LogisticRegressionClassifier : Classifier
    {

        private LogisticRegression logisticRegression;
    
        private double convergenceTolerance;
        
        private int maxIterations;
        public LogisticRegressionClassifier(double ConvergenceTolerance = 1e-6, int MaxIterations = 1000)
            : base(" Logistic Regression ")
        {
                
            convergenceTolerance= ConvergenceTolerance;
            maxIterations = MaxIterations;
            logisticRegression = null ;
        }
        
  
        public double ConvergenceTolerance 
        { 
            get { return convergenceTolerance; }        }
        
        public int MaxIterations 
        { 
            get { return maxIterations; } 
        }

        public override void Train(double[][] features, int[] labels)
        {
            try
            {
                numberOfFeatures= features[0].Length;
                // accord wants true/false instead of 1/0, so convert
                bool[] boolLabels = new bool[labels.Length] ;
                for (int i = 0; i < labels.Length; i ++)
                {
                    boolLabels[i] = labels[i] == 1;
                }
                
                IterativeReweightedLeastSquares<LogisticRegression> teacher = new IterativeReweightedLeastSquares<LogisticRegression>()
                {
                    MaxIterations = maxIterations,
                    Tolerance = convergenceTolerance
                };
                
                // train
                logisticRegression= teacher.Learn(features, boolLabels);
                isTrained = true;
                
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"  training failed: {ex.Message}", ex);
            }
        }
        

        protected override int Decide(double[] features )
        {
            if (!isTrained)
                throw new InvalidOperationException("  Model must be trained before making predictions");
                
            try
            {
                // use accord's decision method (gives true/false, convert to 1/0)
                bool decision = logisticRegression.Decide(features);
                if (decision)
                {
                    return 1;
                }
                else
                {
                    return 0;
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Prediction failed: {ex.Message}", ex);
            }
        }
        
        public override double GetProbability(double[] features)
        {
            if (!isTrained)
                throw new InvalidOperationException("Model must be trained before calculating probabilities");
                
            try
            {
                // using Accord Probability method
                double probability = logisticRegression.Probability(features);
                
                return probability;
            }
            catch (Exception ex )
            {
                throw new InvalidOperationException($"Probability calculation failed: {ex.Message}", ex);
            }
        }
        
        public override double[] GetProbability(double[][] features)
        {
            if (!isTrained)
                throw new InvalidOperationException(" The model must be trained before calculating probabilities" );
                
            try
            {
                // use Accord batch Probability method for efficiency
                double[] probabilities = logisticRegression.Probability(features);
                
                return probabilities;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Batch probability calculation failed: {ex.Message}", ex);
            }
        }
        
        public override void DisplayModelInfo()
        {
            Console.WriteLine("=== Logistic Regression Model Information ===      ");
            Console.WriteLine($"Model Name: {modelName}");
            string  trainingStatus;
            if (isTrained)
            {
                trainingStatus = "Trained" ;
            }
            else
            {
                trainingStatus = "Not Trained!!";
            }
            Console.WriteLine($"Training Status: {trainingStatus}");
            Console.WriteLine($" Decision Threshold: {decisionThreshold:F3}" );
            Console.WriteLine($"Convergence Tolerance: {convergenceTolerance:E}" );
            Console.WriteLine($" maximum Iterations: {maxIterations}");
            
            if (isTrained)
            {
                Console.WriteLine($" number of Features: {numberOfFeatures}");
                Console.WriteLine($"Number of Classes: {numberOfClasses}" );
                
                // display model coefficients 
                if (logisticRegression != null && logisticRegression.Weights != null )
                {
                    Console.WriteLine($"   model Coefficients: {logisticRegression.Weights.Length}");
                    Console.WriteLine($"Intercept: {logisticRegression.Intercept:F6}") ;
                }
            }
            else
            {
                Console.WriteLine("Model not trained - detailed information not available");
            }
            
            Console.WriteLine("============================================");
        }
    }
} 