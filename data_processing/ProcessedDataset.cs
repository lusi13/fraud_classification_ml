using Accord.Statistics.Filters;

namespace FraudDetection.Data
{
    //holds  cleaned data that's ready for ml model
    public class ProcessedDataset
    {
        public double[][] Features { get; set; }
        public int[] Labels { get; set; }
        public string[] FeatureNames { get; set; }
        public Codification Codification { get; set; }
        
        //saved values for normalizing age later
        public double AgeMin { get; set; }
        public double AgeMax { get; set; }

        //  constructor to set default values
        public ProcessedDataset()
        {
            Features= new double[0][] ;
            Labels = new int[0];
            FeatureNames = new string[0];
            Codification = null ; // we'll set this when we encode text features
            AgeMin = 0.0;
            AgeMax = 0.0;
        }
    }
} 