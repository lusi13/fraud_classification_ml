using System;

namespace FraudDetection.Entities
{
    public class InsuranceRecord
    {
        public string Month { get; private set; }
        public string AccidentArea { get ; private set; }
        public string Sex { get; private set  ; }
        public int Age { get; private set; }
        public string Fault { get ; private set; }
        public string PolicyType{ get; private set; }
        public string VehiclePrice { get; private set; }

        //  we're trying to predictif it was fraud(0=no, 1=yes)
        public int FraudFoundP { get ; private set ; }

        public string Make  { get ; private set; }
        public  string Deductible { get; private set; }
        public string DaysPolicyClaim { get ; private set; }
        public string PastNumberOfClaims { get ; private set; }
        public string AgeOfVehicle  { get; private set;   }
        public string PoliceReportFiled { get; private set; }
        public   string WitnessPresent { get; private set; }
        public string AgentType { get; private set; }
        public string NumberOfSuppliments   { get; private set; }
        public string AddressChangeClaim    { get ; private set; }


        public InsuranceRecord(
            string month,
            string accidentArea,
            string sex,
            int  age,
            string fault,
            string policyType,
            string vehiclePrice ,
            int  fraudFoundP,
            string make,
            string deductible,
            string  daysPolicyClaim ,
            string pastNumberOfClaims ,
            string ageOfVehicle,
            string policeReportFiled,
            string   witnessPresent,
            string agentType ,
            string numberOfSuppliments,
            string addressChangeClaim)
        {
            Month = month;
            AccidentArea= accidentArea;
            Sex = sex;
            Fault= fault;
            PolicyType = policyType;
            VehiclePrice = vehiclePrice;
            Make = make;
            DaysPolicyClaim =  daysPolicyClaim;
            PastNumberOfClaims =  pastNumberOfClaims;
            AgeOfVehicle = ageOfVehicle;
            PoliceReportFiled = policeReportFiled;
            WitnessPresent = witnessPresent;
            AgentType = agentType;
            NumberOfSuppliments =  numberOfSuppliments;
            AddressChangeClaim =  addressChangeClaim;
            Deductible =  deductible;

            // check if  numbers make sense
            if (age < 0 || age > 100)
                throw new ArgumentException($" age must be between 0 and 100, got instead: {age}", "age");
            Age = age;

            if (fraudFoundP != 0 && fraudFoundP != 1)
                throw new ArgumentException($"FraudFoundP must be 0 or 1, got: {fraudFoundP}", "fraudFoundP");
            FraudFoundP = fraudFoundP;
        }


        public override string ToString()
        {
            return $"InsuranceRecord [Make: { Make}, Age: {Age}, Fraud: {FraudFoundP}, " +
                   $"AccidentArea: {AccidentArea}]";
        }

        public bool IsFraudulent()
        {
            return FraudFoundP== 1;
        }

    }
} 