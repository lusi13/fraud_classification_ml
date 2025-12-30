using System;
using System.Collections.Generic;
using System.IO;
using FraudDetection.Entities;

namespace FraudDetection.Data
{
    public static class DataLoader
    {
        private const int TotalFields =33;

        public static List<InsuranceRecord> LoadDataset( string filePath)
        {
            if (string.IsNullOrWhiteSpace(filePath))
                throw new ArgumentException("File path cannot be null ");

            if (!File.Exists( filePath))
                throw new FileNotFoundException($"dataset file not found: {filePath}");

            List<InsuranceRecord> records = new List< InsuranceRecord>();
            int lineNumber =0;
            int validRecords = 0 ;
            int skippedRecords = 0;

            try
            {     string[] lines = File.ReadAllLines(filePath);
                
                foreach (string line in lines)
                {
                    lineNumber++;
                    
                    // skip the header line ( has column names)
                    if (lineNumber == 1)
                        continue ;

                    try
                    {
                        string[] fields = line.Split(',');

                        InsuranceRecord record = ParseRecord(fields);
                        if ( IsValidRecord( record))
                        {
                            records.Add(record);
                            validRecords++;
                        }
                        else
                        {
                            skippedRecords++;
                        }
                    }
                    catch
                    {    skippedRecords  ++;  }
                }

                Console.WriteLine($"Dataset loading completed:");
                Console.WriteLine($"  - Total lines processed: {lineNumber - 1}");
                Console.WriteLine($"  - valid records loaded: {validRecords }");
                Console.WriteLine($"  - Records skipped: {skippedRecords}");

                return records;
            }
            catch (Exception ex)
            {
                throw new Exception($"error loading the dataset: {ex.Message} ");
            }
        }

        private static InsuranceRecord ParseRecord(string[] fields)
        {
            // take only the values we need from specific column positions
            // these positions match the csv file structure
            
            string month = fields[0];
            string make =fields[3];
            string accidentArea = fields[4];
            string sex = fields[8];
            string fault = fields[11];
            string policyType =fields[12];
            string vehiclePriceStr = fields[14];
            string deductible = fields[18];
            string daysPolicyClaim =fields[21];
            string pastNumberOfClaims = fields[22];
            string ageOfVehicle= fields[23];
            string policeReportFiled = fields[25];
            string witnessPresent = fields[26];
            string agentType= fields[27];
            string numberOfSuppliments = fields[28];
            string addressChangeClaim= fields[29];

            // convert text numbers to actual numbers
            int age = ParseInt(fields[10]);
            int fraudFoundP = ParseInt(fields[15]);

            return new InsuranceRecord(
                month, accidentArea, sex , age, fault, policyType ,
                vehiclePriceStr , fraudFoundP, make , deductible,  daysPolicyClaim,
                pastNumberOfClaims , ageOfVehicle, policeReportFiled,  witnessPresent,
                agentType, numberOfSuppliments , addressChangeClaim
            );
        }


        private static int ParseInt(string field)
        {
            try
            {
                return int.Parse(field);
            }
            catch
            {
                return 0; // return 0 if it's not a valid number
 }
        }

        private static bool IsValidRecord(InsuranceRecord record)
        {
            // make sure the important fields are not emptyy
            if (string.IsNullOrWhiteSpace(record.Make) ||
                string.IsNullOrWhiteSpace(record.AccidentArea) ||
                string.IsNullOrWhiteSpace(record.Sex) ||
                string.IsNullOrWhiteSpace(record.PolicyType))
            {
                return false;
            }

            return true;
        }
    }
} 