using System;
using System.Collections.Generic;
using System.IO;
using FraudDetection.Data;
using FraudDetection.Entities;

namespace FraudDetection.Tests
{
    public static class DataLoaderTests
    {
        public static void RunTests()
        {
            Console.WriteLine("=== DataLoader Tests ===");
            
            TestLoadDataset();
            TestLoadDatasetWithInvalidPath();
            
            Console.WriteLine("=== Tests Completed ===");
        }

        private static void TestLoadDataset()
        {
            Console.WriteLine("Test: LoadDataset_ShouldLoadRecordsFromCsv" );
            
            try
            {
                string datasetPath = Path.Combine("dataset", "dataset.csv");
                List<InsuranceRecord> records = DataLoader.LoadDataset(datasetPath);
                
                if (records != null && records.Count > 0)
                {
                    InsuranceRecord firstRecord = records[0] ;
                    if (!string.IsNullOrWhiteSpace(firstRecord.Make) && 
                        !string.IsNullOrWhiteSpace(firstRecord.AccidentArea) && 
                        firstRecord.Age > 0 )
                    {
                        Console.WriteLine(" PASSED - Dataset loaded successfully with valid records");
                        Console.WriteLine($"  Records loaded: {records.Count}");
                        Console.WriteLine($"  First record: {firstRecord.Make}, Age: {firstRecord.Age}");
                    }
                    else
                    {
                        Console.WriteLine(" FAILED   Records loaded but data is invalid");
                    }
                }
                else
                {
                    Console.WriteLine(" FAILED - No records loaded");   }
            }
            catch (Exception ex)
            {
                Console.WriteLine($" FAILED - Exception: {ex.Message}") ;
            }
        }

        private static void TestLoadDatasetWithInvalidPath()
        {
            Console.WriteLine("   Test: LoadDataset_WithInvalidPath_ShouldThrowException");
            
            try
            {
                DataLoader.LoadDataset( "nonexistent.csv");
                Console.WriteLine(" FAILED - Expected exception but none was thrown  !!");
            }
            catch (FileNotFoundException)
            {
                Console.WriteLine("PASSED - Correctly threw FileNotFoundException for invalid path ");
            }
            catch (Exception ex)
            {
                Console.WriteLine($" FAILED - wrong exception type: {ex.GetType().Name}");  }
        }
    }
} 