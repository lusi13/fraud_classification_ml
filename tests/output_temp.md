=== Full Machine Learning Pipeline Demo ===
This demo will:
1. Load the insurance fraud dataset
2. Preprocess the data (cleaning, encoding, normalization)
3. Split into train/test sets
4. Train Logistic Regression and Random Forest classifiers
5. Perform cross-validation on training data
6. Evaluate both models on the test set
7. Compare final results

📁 Step 1: Loading Insurance Fraud Dataset...
==================================================
Dataset loading completed:
  - Total lines processed: 15420
  - Valid records loaded: 15420
  - Records skipped: 0
✅ Successfully loaded 15420 insurance records

🔧 Step 2: Data Preprocessing...
==================================================
Starting data preprocessing...
Cleaning data...
Fixed 320 invalid age values
Encoding features...
Checking for constant features...
No constant features found - all features have variation
Normalizing numerical features...
Age normalized: min=16,00, max=80,00, range=64,00
Preprocessing complete: 15420 samples, 59 features
✅ Data preprocessing completed
   Features: 59
   Samples: 15420
   Class distribution:
     Legitimate claims: 14497 (94,0%)
     Fraudulent claims: 923 (6,0%)

✂️ Step 3: Train/Test Split...
==================================================
Split: 12336 training, 3084 test samples
✅ Data split completed
   Training samples: 12336
   Test samples: 3084

🤖 Step 4: Training Machine Learning Models...
==================================================
Training Logistic Regression Classifier...
Logistic Regression training completed successfully.
Number of features: 59
Number of training samples: 12336
✅ Logistic Regression trained - Training Accuracy: 93,95%
Training Random Forest Classifier...
Random Forest training completed successfully.
Number of features: 59
Number of training samples: 12336
Number of trees: 100
Variables considered per split: 7
✅ Random Forest trained - Training Accuracy: 94,56%

🔄 Step 5: Cross-Validation Evaluation...
==================================================
Evaluating Logistic Regression with 5-fold Cross-Validation:
=== 5-Fold Cross-Validation ===
Total samples: 12336
Shuffle data: True
Random seed: 42

Fold 1/5:
    Fold 1: Removing 1 constant feature(s) from this fold
Logistic Regression training completed successfully.
Number of features: 58
Number of training samples: 9868
  Training samples: 9868
  Test samples: 2468
  Test accuracy: 93,60%

Fold 2/5:
Logistic Regression training completed successfully.
Number of features: 59
Number of training samples: 9869
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 94,20%

Fold 3/5:
Logistic Regression training completed successfully.
Number of features: 59
Number of training samples: 9869
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 94,08%

Fold 4/5:
Logistic Regression training completed successfully.
Number of features: 59
Number of training samples: 9869
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 94,16%

Fold 5/5:
    Fold 5: Removing 1 constant feature(s) from this fold
Logistic Regression training completed successfully.
Number of features: 58
Number of training samples: 9869
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 93,84%

=== Cross-Validation Summary ===
Mean Test Accuracy: 93,98% ± 0,25%
Mean Training Accuracy: 93,96% ± 0,06%
Best Fold: 2 (Accuracy: 94,20%)
Worst Fold: 1 (Accuracy: 93,60%)
===============================
Evaluating Random Forest with 5-fold Cross-Validation:
=== 5-Fold Cross-Validation ===
Total samples: 12336
Shuffle data: True
Random seed: 42

Fold 1/5:
    Fold 1: Removing 1 constant feature(s) from this fold
Random Forest training completed successfully.
Number of features: 58
Number of training samples: 9868
Number of trees: 100
Variables considered per split: 7
  Training samples: 9868
  Test samples: 2468
  Test accuracy: 94,04%

Fold 2/5:
Random Forest training completed successfully.
Number of features: 59
Number of training samples: 9869
Number of trees: 100
Variables considered per split: 7
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 94,57%

Fold 3/5:
Random Forest training completed successfully.
Number of features: 59
Number of training samples: 9869
Number of trees: 100
Variables considered per split: 7
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 94,33%

Fold 4/5:
Random Forest training completed successfully.
Number of features: 59
Number of training samples: 9869
Number of trees: 100
Variables considered per split: 7
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 94,53%

Fold 5/5:
    Fold 5: Removing 1 constant feature(s) from this fold
Random Forest training completed successfully.
Number of features: 58
Number of training samples: 9869
Number of trees: 100
Variables considered per split: 7
  Training samples: 9869
  Test samples: 2467
  Test accuracy: 93,84%

=== Cross-Validation Summary ===
Mean Test Accuracy: 94,26% ± 0,31%
Mean Training Accuracy: 94,61% ± 0,10%
Best Fold: 2 (Accuracy: 94,57%)
Worst Fold: 5 (Accuracy: 93,84%)
===============================
📊 Step 6: Final Test Set Evaluation...
==================================================
Logistic Regression Test Results:
  Test Accuracy: 93,71%
Random Forest Test Results:
  Test Accuracy: 94,00%

📈 Step 7: Comprehensive Results Analysis...
==================================================
CROSS-VALIDATION COMPARISON:
Logistic Regression CV: 93,98% ± 0,25%
Random Forest CV:       94,26% ± 0,31%

TEST SET PERFORMANCE:
Logistic Regression: 93,71%
Random Forest:       94,00%

=== Logistic Regression Performance Metrics ===

📊 CORE METRICS:
┌─────────────┬─────────┬─────────────────────────────────┐
│   Metric    │  Value  │           Interpretation        │
├─────────────┼─────────┼─────────────────────────────────┤
│ Accuracy    │  93,7% │ Overall correctness             │
│ Precision   │   0,0% │ Fraud predictions reliability   │
│ Recall      │   0,0% │ Fraud detection completeness    │
│ F1-Score    │   0,0% │ Balanced precision/recall       │
│ Specificity │  99,9% │ Legitimate detection rate       │
└─────────────┴─────────┴─────────────────────────────────┘

🔢 CONFUSION MATRIX:
                    PREDICTED
                 Legit    Fraud
       ┌─────────────────────────┐
  Legit│  2890        4   │
ACTUAL │                         │
 Fraud │   190        0   │
       └─────────────────────────┘

💼 BUSINESS IMPACT:
  • Total test cases: 3.084
  • Actual fraud cases: 190 (6,2%)
  • Actual legitimate cases: 2.894 (93,8%)

  • Correctly identified fraud: 0 out of 190 (0,0%)
  • Missed fraud cases: 190 (could cost money!)
  • False alarms: 4 (unnecessary investigations)
  • Correctly cleared legitimate: 2.890 out of 2.894

🎯 PROBABILITY ANALYSIS:
  • Average fraud probability for actual fraud cases: 0,131
  • Average fraud probability for legitimate cases: 0,057
  • Probability separation: 0,075

🔍 MODEL INTERPRETATION:
  ⚠️  Model is too conservative - doesn't predict any fraud!
      This minimizes false alarms but misses all fraud cases.

════════════════════════════════════════════════════════════

=== Random Forest Performance Metrics ===

📊 CORE METRICS:
┌─────────────┬─────────┬─────────────────────────────────┐
│   Metric    │  Value  │           Interpretation        │
├─────────────┼─────────┼─────────────────────────────────┤
│ Accuracy    │  94,0% │ Overall correctness             │
│ Precision   │ 100,0% │ Fraud predictions reliability   │
│ Recall      │   2,6% │ Fraud detection completeness    │
│ F1-Score    │   5,1% │ Balanced precision/recall       │
│ Specificity │ 100,0% │ Legitimate detection rate       │
└─────────────┴─────────┴─────────────────────────────────┘

🔢 CONFUSION MATRIX:
                    PREDICTED
                 Legit    Fraud
       ┌─────────────────────────┐
  Legit│  2894        0   │
ACTUAL │                         │
 Fraud │   185        5   │
       └─────────────────────────┘

💼 BUSINESS IMPACT:
  • Total test cases: 3.084
  • Actual fraud cases: 190 (6,2%)
  • Actual legitimate cases: 2.894 (93,8%)

  • Correctly identified fraud: 5 out of 190 (2,6%)
  • Missed fraud cases: 185 (could cost money!)
  • False alarms: 0 (unnecessary investigations)
  • Correctly cleared legitimate: 2.894 out of 2.894

🎯 PROBABILITY ANALYSIS:
  • Average fraud probability for actual fraud cases: 0,051
  • Average fraud probability for legitimate cases: 0,009
  • Probability separation: 0,042

🔍 MODEL INTERPRETATION:
  🎯 Model is conservative - high precision, low recall
      When it predicts fraud, it's usually right, but misses many cases.

════════════════════════════════════════════════════════════

🔄 === COMPREHENSIVE MODEL COMPARISON === 🔄

📊 TEST SET PERFORMANCE COMPARISON:
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│     Metric      │ Logistic Regres│  Random Forest  │   Best Model    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Accuracy        │       93,7%     │       94,0%     │   Random Forest │
│ Precision       │        0,0%     │      100,0%     │   Random Forest │
│ Recall          │        0,0%     │        2,6%     │   Random Forest │
│ F1-Score        │        0,0%     │        5,1%     │   Random Forest │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

🔄 CROSS-VALIDATION PERFORMANCE:
┌─────────────────┬─────────────────┬─────────────────┬─────────────────┐
│   CV Metric     │ Logistic Regres│  Random Forest  │   Best Model    │
├─────────────────┼─────────────────┼─────────────────┼─────────────────┤
│ Mean Accuracy   │       94,0%     │       94,3%     │   Random Forest │
│ Std Deviation   │        0,3%     │        0,3%     │    Logistic Reg │
│ Best Fold       │       94,2%     │       94,6%     │   Random Forest │
└─────────────────┴─────────────────┴─────────────────┴─────────────────┘

🕵️ FRAUD DETECTION EFFECTIVENESS:
┌─────────────────┬─────────────────┬─────────────────┐
│    Outcome      │ Logistic Regres│  Random Forest  │
├─────────────────┼─────────────────┼─────────────────┤
│ Fraud Detected  │         0       │         5       │
│ Fraud Missed    │       190       │       185       │
│ False Alarms    │         4       │         0       │
│ Legit Cleared   │     2.890       │     2.894       │
└─────────────────┴─────────────────┴─────────────────┘

💰 BUSINESS IMPACT ANALYSIS:
  Total fraud cases in test set: 190

  Logistic Regression:
    • Caught 0 fraud cases (0,0% detection rate)
    • Missed 190 fraud cases (potential losses)
    • Generated 4 false alarms (investigation costs)

  Random Forest:
    • Caught 5 fraud cases (2,6% detection rate)
    • Missed 185 fraud cases (potential losses)
    • Generated 0 false alarms (investigation costs)