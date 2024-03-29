# WHY FFORMA predictor

[![Linter][linter-image]][linter-url]
[![Unit-Tests][unit-tests-image]][unit-tests-url]
[![Integration-Tests][integration-tests-image]][integration-tests-url]

[linter-image]: https://github.com/DeustoTech/why_predictor/actions/workflows/linter.yml/badge.svg?branch=master
[linter-url]: https://github.com/DeustoTech/why_predictor/actions/workflows/linter.yml
[unit-tests-image]: https://github.com/DeustoTech/why_predictor/actions/workflows/unit-tests.yml/badge.svg?branch=master
[unit-tests-url]: https://github.com/DeustoTech/why_predictor/actions/workflows/unit-tests.yml
[integration-tests-image]: https://github.com/DeustoTech/why_predictor/actions/workflows/integration-tests.yml/badge.svg?branch=master
[integration-tests-url]: https://github.com/DeustoTech/why_predictor/actions/workflows/integration-tests.yml

## install requirements
`pip install -r requirements.txt`

## execute
`python -m why_predictor`

## Configuration
Edit the configuration file to modify the parameters or include the corresponding command line parameter to change it (just for that execution).
For more information, execute:

```ShellSession
usage: python -m why_predictor [-h] [-v] [-m {generate-csvs,
    generate-hyperparams,generate-fforma,evaluate-fforma,full}]
       [--base-path-dataset DATASET_BASEPATH]
       [--dataset-dir-name DATASET_DIR_NAME]
       [--window-num-features NUM_FEATURES]
       [--window-num-predictions NUM_PREDICTIONS]
       [--save-datasets SAVE_DATASETS]
       [--njobs NJOBS]
       [--use-models-training {SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,SHIFT_SVR,SHIFT_SGD,
           SHIFT_MLP,CHAIN_LR,CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,CHAIN_MLP,
           MULTI_LR,MULTI_RF,MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP}
           [{SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,SHIFT_SVR,SHIFT_SGD,SHIFT_MLP,CHAIN_LR,
             CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,CHAIN_MLP,MULTI_LR,MULTI_RF,
             MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP}
           ...]]
       [--error-type-training {MAPE,MAE,RMSE,SMAPE}]
       [--percentage-csv-files-for-training-hyperparameters TRAINING_PERCENTAGE_HYPERPARAMS]
       [--train-test-ratio-hyperparameters TRAIN_TEST_RATIO_HYPERPARAMS |
        --initial-training-path INITIAL_TRAINING_PATH]
       [--use-models-fforma {SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,SHIFT_SVR,SHIFT_SGD,SHIFT_MLP,
           CHAIN_LR,CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,CHAIN_MLP,MULTI_LR,MULTI_RF,
           MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP}
           [{SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,SHIFT_SVR,SHIFT_SGD,SHIFT_MLP,CHAIN_LR,
             CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,CHAIN_MLP,MULTI_LR,MULTI_RF,
             MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP}
           ...]]
       [--error-type-fforma {MAPE,MAE,RMSE,SMAPE}]
       [--percentage-csv-files-for-training-fforma TRAINING_PERCENTAGE_FFORMA]
       [--train-test-ratio-fforma TRAIN_TEST_RATIO_FFORMA]
       [--percentage-csv-files-for-fforma-eval TRAINING_PERCENTAGE_FFORMA_EVAL]
       [--train-test-ratio-fforma-eval TRAIN_TEST_RATIO_FFORMA_EVAL]
       [--error-type-fforma-eval {MAPE,MAE,RMSE,SMAPE}]
       [--use-fforms]

WHY Predictor

options:
  -h, --help            show this help message and exit
  -v, --verbose
  -m {generate-csvs,generate-hyperparams,generate-fforma,evaluate-fforma,full},
  --mode {generate-csvs,generate-hyperparams,generate-fforma,evaluate-fforma,full}
                        Select the operation mode, by default it will run in full mode that
                        includes both generate-errors and generate fforma. generate-errors:
                        will only train the models to generate the error files, while
                        generate-fforma will assume the hyperparameters are already set, so
                        it will generate the FFORMA model.
  --base-path-dataset DATASET_BASEPATH
                        base path where dataset are stored
  --dataset-dir-name DATASET_DIR_NAME
                        exact name of the directory containing the CSV files
  --window-num-features NUM_FEATURES
                        num of hours used as features
  --window-num-predictions NUM_PREDICTIONS
                        num of hours used as predictions
  --save-datasets SAVE_DATASETS
                        save generated rolling-window datasets to disk
  --njobs NJOBS         Number of CPUs to use. When negative values are provided, -1 means
                        all CPUs, -2: means all CPUs but one, -3: means all CPUs but two...

Model training:
  --use-models-training {SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,SHIFT_SVR,SHIFT_SGD,SHIFT_MLP,
      CHAIN_LR,CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,CHAIN_MLP,MULTI_LR,MULTI_RF,
      MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP} [{SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,
      SHIFT_SVR,SHIFT_SGD,SHIFT_MLP,CHAIN_LR,CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,
      CHAIN_MLP,MULTI_LR,MULTI_RF,MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP} ...]
                        Select what models to use:
                            SHIFT_LR (Shifted Linear Regression)
                            SHIFT_RF (Shifted Random Forest Regression)
                            SHIFT_KNN (Shifted KNN Regression)
                            SHIFT_DT (Shifted Decission Tree Regression)
                            SHIFT_SVR (Shifted Support Vector Regression)
                            SHIFT_SGD (Shifted Stochastic Gradient Descent Regressor)
                            SHIFT_MLP (Shifted Multi-layer Perceptron Regressor)
                            CHAIN_LR (Chained Linear Regression)
                            CHAIN_RF (Chained Random Forest Regression)
                            CHAIN_KNN (Chained KNN Regression)
                            CHAIN_DT (Chained Decission Tree Regression)
                            CHAIN_SVR (Chained Support Vector Regression)
                            CHAIN_SGD (Chained Stochastic Gradient Descent Regressor)
                            CHAIN_MLP (Chained Multi-layer Perceptron Regressor)
                            MULTI_LR (Multioutput Linear Regression)
                            MULTI_RF (Multioutput Random Forest Regression)
                            MULTI_KNN (Multioutput KNN Regression)
                            MULTI_DT (Multioutput Decission Tree Regression)
                            MULTI_SVR (Multioutput Support Vector Regression)
                            MULTI_SGD (Multioutput SGD Regression)
                            MULTI_MLP (Multioutput MLP Regression)
  --error-type-training {MAPE,MAE,RMSE,SMAPE}
                        metric to calculate the error
  --percentage-csv-files-for-training-hyperparameters TRAINING_PERCENTAGE_HYPERPARAMS
                        Percentage of the CSV files that will be used for training
  --train-test-ratio-hyperparameters TRAIN_TEST_RATIO_HYPERPARAMS
                        ratio of samples used for training
                        (1 - this value will be used for testing)
  --initial-training-path INITIAL_TRAINING_PATH
                        path to a folder where datasets will be used just for training of models
                        in phase1 (if this mode if used)

FFORMA training:
  --use-models-fforma {SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,SHIFT_SVR,SHIFT_SGD,SHIFT_MLP,
      CHAIN_LR,CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,CHAIN_MLP,MULTI_LR,MULTI_RF,
      MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP} [{SHIFT_LR,SHIFT_RF,SHIFT_KNN,SHIFT_DT,
      SHIFT_SVR,SHIFT_SGD,SHIFT_MLP,CHAIN_LR,CHAIN_RF,CHAIN_KNN,CHAIN_DT,CHAIN_SVR,CHAIN_SGD,
      CHAIN_MLP,MULTI_LR,MULTI_RF,MULTI_KNN,MULTI_DT,MULTI_SVR,MULTI_SGD,MULTI_MLP} ...]
                        Select what models to use:
                            MULTI_LR (Multioutput Linear Regression)
                            MULTI_RF (Multioutput Random Forest Regression)
                            MULTI_KNN (Multioutput KNN Regression)
                            MULTI_DT (Multioutput Decission Tree Regression)
                            MULTI_SVR (Multioutput Support Vector Regression)
                            MULTI_SGD (Multioutput SGD Regression)
                            MULTI_MLP (Multioutput MLP Regression)
  --error-type-fforma {MAPE,MAE,RMSE,SMAPE}
                        metric to calculate the error
  --percentage-csv-files-for-training-fforma TRAINING_PERCENTAGE_FFORMA
                        Percentage of the CSV files that will be used for training
  --train-test-ratio-fforma TRAIN_TEST_RATIO_FFORMA
                        ratio of samples used for training
                        (1 - this value will be used for testing)

FFORMA evaluation:
  --percentage-csv-files-for-fforma-eval TRAINING_PERCENTAGE_FFORMA_EVAL
                        Percentage of the CSV files that will be used for evaluation
  --train-test-ratio-fforma-eval TRAIN_TEST_RATIO_FFORMA_EVAL
                        ratio of samples used for evaluation
                        (1 - this value will be used for evaluation)
  --error-type-fforma-eval {MAPE,MAE,RMSE,SMAPE}
                        metric to calculate the error when evaluating the final output of FFORMA
  --use-fforms          use this flag if you want to execute FFORMS instead of FFORMA
```
