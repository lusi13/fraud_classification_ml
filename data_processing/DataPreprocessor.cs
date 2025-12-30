using System;
using System.Collections.Generic;
using System.Linq;
using Accord.Statistics.Filters;
using Accord.Statistics;
using FraudDetection.Entities;

namespace FraudDetection.Data
{

    // cleans data, turns text into numbers, and makes features similar sizes
    public static class DataPreprocessor
    {
        // default values for fixing bad data
        private const int DefaultAge = 40;
        private const int MinValidAge = 16;
        private const int MaxValidAge = 100;

        // main method          does all the data preparation steps
        public static ProcessedDataset PreprocessData(List<InsuranceRecord> records)
        {
            if (records == null || records.Count == 0)
                throw new ArgumentException("Records cannot be null or empty");

            Console.WriteLine("Starting data preprocessing  ....");
            
            ProcessedDataset result = new ProcessedDataset();

            List<InsuranceRecord> cleanedRecords = CleanData(records);
            
            result = EncodeFeatures(cleanedRecords);
            result.Labels = ExtractLabels(cleanedRecords);

            // remove features that are always the same (useless for learning)
            result = CheckAndHandleConstantFeatures(result);

            // make all numbers similar sizes (helps the algorithm learn better)
            result = NormalizeNumericalFeatures(result);

            Console.WriteLine($"Preprocessing complete: {result.Features.Length} samples, {result.Features[0].Length} features");
            return result;
        }

        // fixes ages that don't make sense (like negative ages or 200 years old)
        private static List<InsuranceRecord> CleanData(List<InsuranceRecord> records)
        {
            Console.WriteLine("Cleaning data..");
            
            List<InsuranceRecord> cleanedRecords = new List<InsuranceRecord>();
            int fixedAgeCount = 0;

            foreach (InsuranceRecord record in records)
            {
                InsuranceRecord cleanedRecord = record;
                
                // if age is weird, replace it with a normal age (40)
                if (record.Age < MinValidAge || record.Age > MaxValidAge)
                {
                    cleanedRecord = new InsuranceRecord(
                        record.Month, record.AccidentArea, record.Sex, DefaultAge,
                        record.Fault, record.PolicyType, record.VehiclePrice, record.FraudFoundP,
                        record.Make, record.Deductible, record.DaysPolicyClaim,
                        record.PastNumberOfClaims, record.AgeOfVehicle, record.PoliceReportFiled,
                        record.WitnessPresent, record.AgentType, record.NumberOfSuppliments, 
                        record.AddressChangeClaim
                    );
                    fixedAgeCount++;
                }
                
                cleanedRecords.Add(cleanedRecord);
            }

            Console.WriteLine($" fixed {fixedAgeCount} invalid age values");
            return cleanedRecords;
        }

        private static int[] ExtractLabels(List<InsuranceRecord> records)
        {
            int[] labels = new int[records.Count];
            for (int i = 0; i < records.Count; i++)
            {
                labels[i] = records[i].FraudFoundP;
            }
            return labels;
        }

        // turns text features into numbers
        // simple ones (yes/no) become 0 or 1, complex ones (like car brands) get one  hot encoding
        private static ProcessedDataset EncodeFeatures(List<InsuranceRecord> records)
        {
            Console.WriteLine("Encoding features...");
            
            ProcessedDataset result = new ProcessedDataset();
            List<string> allFeatureNames = new List<string>();

            double[][] binaryFeatures = EncodeBinaryFeatures(records);
            allFeatureNames.AddRange(new[] { 
                "AccidentArea_Urban", "Sex_Male", "Fault_PolicyHolder", 
                "PoliceReportFiled_Yes", "WitnessPresent_Yes", "AgentType_Internal",
                "VehiclePrice_NormalRange" 
            });

            double[][] numericFeatures = ExtractNumericFeatures(records);
            allFeatureNames.AddRange(new[] { "Age" });

            // handle complex features (like car brands) with onehot encoding
            ProcessedDataset oneHotResult = EncodeOneHotFeatures(records);
            allFeatureNames.AddRange(oneHotResult.FeatureNames);

            result.Features = CombineFeatures(binaryFeatures, numericFeatures, oneHotResult.Features);
            result.Codification = oneHotResult.Codification;
            result.FeatureNames = allFeatureNames.ToArray();

            return result;
        }

        // turns simple yes/no features into 1s and 0s
        private static double[][] EncodeBinaryFeatures( List<InsuranceRecord> records)
        {
            double[][] features = new double[records.Count][];
            
            for (int i = 0; i < records.Count; i++)
            {
                InsuranceRecord record = records[i];
                features[i] = new double[7]; 

                // urban area = 1, rural = 0
                if (record.AccidentArea.ToLower() == "urban")
                {
                    features[i][0] = 1.0;
                }
                else
                {
                    features[i][0] = 0.0 ;   }

                // male= 1, female = 0
                if (record.Sex.ToLower() == "male")
                {     features[i][1] = 1.0;    }
                
                else
                { features[i][1] = 0.0; }

                // policy holder at fault= 1, third party = 0
                if (record.Fault.ToLower() == " policy holder ")
                {
                    features[i][2] = 1.0;
                }
                else
                {
                    features[i][2] = 0.0;
                }

                // police report filed= 1, no report = 0
                if (record.PoliceReportFiled.ToLower() == "yes")
                {
                    features[i][3] = 1.0;
                }
                else
                {   features[i][3] = 0.0 ; }

                // witness present = 1, no witness = 0
                if (record.WitnessPresent.ToLower() == "yes")
                {
                    features[i][4] = 1.0;
                }
                else
                {
                    features[i][4] = 0.0;
                }

                // internal agent = 1, external = 0
                if (record.AgentType.ToLower() == "internal")

                {  features[i][5] = 1.0 ;  }
                else
                {   features[i][5] = 0.0;  }

                // price in normal range = 0, extreme prices (very cheap/very expensive) = 1
                string priceCategory = record.VehiclePrice.ToLower().Trim();
                if (priceCategory == "less than 20000" || priceCategory == "more than 69000")
                {
                    features[i][6] = 1.0; // extreme prices
                }
                else
                {
                    features[i][6] = 0.0; // normal price ranges (20k-69k)
                }
            }

            return features;
        }

        private static double[][] ExtractNumericFeatures(List<InsuranceRecord> records)
        {
            double[][] features = new double[records.Count][];
            
            for (int i = 0; i < records.Count; i++)
            {
                features[i] = new double[1];
                features[i][0] = records[i].Age;
            }

            return features;
        }

        // handles  features with many categories (like car make)
        // creates separate columns for each possible value
        private static ProcessedDataset EncodeOneHotFeatures( List<InsuranceRecord> records)
        {
            // these features have  different values
            string[] columnNames = new[] { "Make", "Month", "PolicyType" , "AgeOfVehicle", "Deductible" , 
                                          "DaysPolicyClaim", "PastNumberOfClaims" , "NumberOfSuppliments", "AddressChangeClaim" };
            
            string[][] categoricalData = new string[ records.Count][];
            for (int i = 0; i < records.Count ; i ++)
            {
                InsuranceRecord record = records[i];
                categoricalData[i] = new[] {
                    record.Make,
                    record.Month,
                    record.PolicyType ,
                    record.AgeOfVehicle,
                    record.Deductible ,
                    record.DaysPolicyClaim,
                    record.PastNumberOfClaims ,
                    record.NumberOfSuppliments,
                    record.AddressChangeClaim
                };
            }

            // use accord's onehot encoder
            Codification codification = new Codification(columnNames, categoricalData);
            int[][] codes = codification.Transform(categoricalData);

            int totalFeatures = 0;
            for (int col = 0; col < columnNames.Length; col++)
            {
                totalFeatures += codification[columnNames[col]].NumberOfSymbols;
            }

            double[][] features = new double[records.Count][];
            for (int i = 0; i < records.Count; i++)
            {
                features[i] = new double[totalFeatures];
                int featureIndex = 0;

                for (int col = 0; col < columnNames.Length; col++)
                {
                    int categoryCount = codification[columnNames[col]].NumberOfSymbols;
                    int codeValue = codes[i][col];
                    
                    if (codeValue >= 0 && codeValue < categoryCount)
                    {
                        features[i][featureIndex + codeValue] = 1.0;
                    }
                    
                    featureIndex += categoryCount;
                }
            }

            // just use numbers instead of trying to get actual names
            List<string> names = new List<string>();
            foreach (string col in columnNames)
            {
                for (int sym = 0; sym < codification[col].NumberOfSymbols; sym++)
                {
                    names.Add($"{col}_{sym}") ;
                }
            }

            return new ProcessedDataset
            {
                Features = features,
                Codification = codification,
                FeatureNames = names.ToArray()
            };
        }

        private static double[][] CombineFeatures(double[][] binaryFeatures, double[][] numericFeatures, double[][] oneHotFeatures)
        {
            int recordCount = binaryFeatures.Length;
            int totalFeatures = binaryFeatures[0].Length + numericFeatures[0].Length + oneHotFeatures[0].Length;
            
            double[][] combinedFeatures = new double[recordCount][];
            
            for (int i = 0; i < recordCount; i++)
            {
                double[] combined = new double[totalFeatures];
                int index = 0;

                Array.Copy(binaryFeatures[i], 0, combined, index, binaryFeatures[i].Length);
                index += binaryFeatures[i].Length;

                Array.Copy(numericFeatures[i], 0, combined, index, numericFeatures[i].Length);
                index += numericFeatures[i].Length;

                Array.Copy(oneHotFeatures[i], 0, combined, index, oneHotFeatures[i].Length);

                combinedFeatures[i] = combined;
            }

            return combinedFeatures;
        }

        // splits data into training and testing sets
        // shuffles everything first so that it's random
        public static (double[][] trainFeatures, int[] trainLabels, double[][] testFeatures, int[] testLabels) 
            SplitTrainTest(double[][] features, int[] labels, double testRatio = 0.2)
        {
            if (features.Length != labels.Length)
                throw new ArgumentException("Features and labels must have same length");

            // make a list of row numbers and shuffle them randomly
            int[] indices = Enumerable.Range(0, features.Length).ToArray();
            
            // use same random seed per la   riproducibilità dei results
            Random seededRandom = new Random(42);
            seededRandom.Shuffle(indices);

            int testSize = (int)(features.Length * testRatio);
            int trainSize = features.Length - testSize;

            double[][] trainFeatures = new double[trainSize][];
            int[] trainLabels = new int[trainSize];
            double[][] testFeatures = new double[testSize][];
            int[] testLabels = new int[testSize];

            for (int i = 0; i < trainSize; i++)
            {
                int originalIndex = indices[i];
                trainFeatures[i] = features[originalIndex];
                trainLabels[i] = labels[originalIndex];
            }

            for (int i = 0; i < testSize; i++)
            {
                int originalIndex = indices[trainSize + i];
                testFeatures[i] = features[originalIndex];
                testLabels[i] = labels[originalIndex];
            }

            Console.WriteLine($"Split: {trainSize} training, {testSize} test samples");
            return (trainFeatures, trainLabels, testFeatures, testLabels);
        }

        // makes age values between 0 and 1 so the algorithm works better
        // finds the smallest and biggest age then scales everything proportionally
        private static ProcessedDataset NormalizeNumericalFeatures(ProcessedDataset dataset)
        {
            Console.WriteLine(" Normalizing numerical features.....");
            
            int ageFeatureIndex = FindFeatureIndex(dataset.FeatureNames, "Age");
            
            if (ageFeatureIndex == -1)
            {
                Console.WriteLine(" age feature not found, skipping normalization");
                return dataset;
            }
            
            double minAge = double.MaxValue;
            double maxAge = double.MinValue;
            
            for (int i = 0; i < dataset.Features.Length; i ++ )
            {
                double ageValue = dataset.Features[i][ageFeatureIndex];
                if (ageValue < minAge)
                    minAge = ageValue;
                if (ageValue > maxAge)
                    maxAge = ageValue;
            }
            
            dataset.AgeMin = minAge ;
            dataset.AgeMax =  maxAge;
            
            // scale all ages to be between 0 and 1
            double range= maxAge - minAge;
            if (range > 0) // make sure to don't divide by 0
            {
                for (int i = 0; i < dataset.Features.Length; i ++)
                {
                    double originalAge = dataset.Features[i][ageFeatureIndex];
                    dataset.Features[i][ageFeatureIndex] = (originalAge - minAge) /range;
                }
            }
            
            Console.WriteLine($"Age normalized: min={minAge:F2}, max ={maxAge:F2}, range={range:F2}");
            return dataset;
        }

        private static int FindFeatureIndex(string[] featureNames, string targetFeatureName)
        {
            for (int i = 0; i < featureNames.Length; i ++)
            {
                if (featureNames[i].Equals(targetFeatureName, StringComparison.OrdinalIgnoreCase))
                {
                    return i;
                }
            }
            return -1;
        }

        // scales a single age value using the same rules we used for training
        // useful when we get new data to predict
        public static double NormalizeAge(double age, double minAge, double maxAge)
        {
            double range = maxAge - minAge;
            if (range <= 0)
                return 0.0;
            
            return (age- minAge) / range;
        }

        public static double DenormalizeAge(double normalizedAge, double minAge , double maxAge)
        {
            double range= maxAge - minAge;
            return normalizedAge * range +minAge;
        }

        // looks for features that are always the same value and removes them
        //  useless features  because they don't help predict anything
        private static ProcessedDataset CheckAndHandleConstantFeatures(ProcessedDataset  dataset)
        {
            Console.WriteLine(" checking for constant features.....");
            
            if (dataset.Features.Length == 0 || dataset.Features[0].Length == 0)
            {
                Console.WriteLine("No features to check!! ");
                return dataset;
            }

            List<int> constantFeatureIndices = new List<int>();
            List<string> constantFeatureNames = new List<string>() ;

            for (int featureIndex = 0; featureIndex < dataset.Features[0].Length; featureIndex ++)
            {
                double[] featureValues = new double[dataset.Features.Length];
                for (int sampleIndex = 0; sampleIndex < dataset.Features.Length; sampleIndex++)
                {
                    featureValues[sampleIndex] = dataset.Features[sampleIndex][featureIndex];
                }
                
                // use accord's math to check if there is  some  variation
                double variance = Measures.Variance(featureValues);
                
                if (variance == 0.0) // no variation = constant feature
                {
                    constantFeatureIndices.Add(featureIndex);
                    string featureName;
                    if (featureIndex < dataset.FeatureNames.Length)
                    {
                        featureName =dataset.FeatureNames[featureIndex];
                    }
                    else
                    {
                        featureName = $" feature_{featureIndex}";
                    }
                    constantFeatureNames.Add(featureName);
                    
                    double constantValue = dataset.Features[0][featureIndex];
                    Console.WriteLine($" constant feature detected: {featureName} (index {featureIndex}) = {constantValue} (variance = 0)");
                }
            }

            if (constantFeatureIndices.Count > 0)
            {
                Console.WriteLine($" removing {constantFeatureIndices.Count} constant features: {string.Join(", ", constantFeatureNames)}");
                dataset = RemoveFeaturesByIndices(dataset, constantFeatureIndices);
                Console.WriteLine($"  After removing constant features: {dataset.Features[0].Length} features remaining");
            }
            else
            {
                Console.WriteLine("No constant features found - all features have variation");
            }

            return dataset;
        }

        private static ProcessedDataset RemoveFeaturesByIndices( ProcessedDataset dataset, List<int> indicesToRemove)
        {
            double[][] newFeatures = new double[dataset.Features.Length][];
            for (int sampleIndex = 0; sampleIndex < dataset.Features.Length; sampleIndex++)
            {
                List<double> newSampleFeatures = new List< double>();
                for (int featureIndex = 0; featureIndex < dataset.Features[sampleIndex].Length; featureIndex++)
                {
                    if (!indicesToRemove.Contains(featureIndex))
                    {
                        newSampleFeatures.Add(dataset.Features[sampleIndex][featureIndex]);
                    }
                }
                newFeatures[sampleIndex] = newSampleFeatures.ToArray();
            }

            List<string> newFeatureNames = new List<string>();
            for (int featureIndex = 0; featureIndex < dataset.FeatureNames.Length ; featureIndex++)
            {
                if (!indicesToRemove.Contains(featureIndex))
                {
                    newFeatureNames.Add(dataset.FeatureNames[featureIndex ]);
                }
            }

            dataset.Features = newFeatures;
            dataset.FeatureNames = newFeatureNames.ToArray();
            
            return dataset ;
        }
    }
} 