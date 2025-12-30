using System;

namespace FraudDetection.ML.Interfaces
{
    // defines what methods all our ml models need to have
    public interface IClassifier
    {
        
        string ModelName { get; }
        bool IsTrained { get; }
        
        // the cutoff point for deciding fraud vs not fraud
        // if probability > this number, we call it fraud
        double DecisionThreshold { get; set; }
        
        
        void Train(double[][] features, int[] labels);
        int Predict(double[] features);
        int[] Predict(double[][] features);
        double GetProbability(double[] features);
        double[] GetProbability(double[][] features);
        
        
        double GetAccuracy(double[][] testFeatures, int[] testLabels);
        void DisplayModelInfo();
        
    }
} 