import os
import optuna
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, logging, DataCollatorWithPadding
from sklearn.metrics import f1_score
from utils import make_current_datetime_dir, preprocess_data, compute_metrics


# Assuming these are already set
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
logging.set_verbosity_error()


import sys
sys.path.append(os.path.abspath(".."))
from constants import ID2LABEL_KOR, ID2LABEL_EN 
LABEL2ID_EN = {v: k for k, v in ID2LABEL_EN.items()}


# Objective function for Optuna optimization
def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 2e-6, 1e-4)
    per_device_train_batch_size = trial.suggest_categorical('per_device_train_batch_size', [8, 16, 32])
    per_device_eval_batch_size = trial.suggest_categorical('per_device_eval_batch_size', [8, 16])
    num_train_epochs = trial.suggest_int('num_train_epochs', 3, 10)
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.1)

    # Define paths to the datasets
    opt = {
        'pretrained_model': 'beomi/KcELECTRA-base-v2022',
        'pretrained_tokenizer': 'beomi/KcELECTRA-base-v2022',
        'problem_type': 'multi_label_classification',
        'train_dataset_path': './data/preprocess/DVforEC(a_6l)_train.csv',
        'val_dataset_path': './data/preprocess/DVforEC(a_6l)_val.csv',
        'output_dir': './weights/',
        'metric_for_best_model': 'f1',
        'eval_strategy': 'epoch',
        'save_strategy': 'epoch',
        'eval_steps': 500,
        'seed': 1031,
        'no_cuda': False,
        'dataloader_num_workers': 4,
        'load_best_model_at_end': False,
        'learning_rate': learning_rate,
        'per_device_train_batch_size': per_device_train_batch_size,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'num_train_epochs': num_train_epochs,
        'weight_decay': weight_decay,
    }

    # Tokenizer and model setup
    tokenizer = AutoTokenizer.from_pretrained(opt['pretrained_tokenizer'])
    id2label = ID2LABEL_EN
    label2id = {v: k for k, v in id2label.items()}
    labels = list(label2id.keys())

    # Load datasets
    train_df = pd.read_csv(opt['train_dataset_path'], dtype=str)
    val_df = pd.read_csv(opt['val_dataset_path'], dtype=str)

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    dataset = DatasetDict({
        'train': train_dataset,
        'val': val_dataset
    })

    # Data preprocessing step
    dataset = dataset.map(
        preprocess_data,
        batched=True,
        remove_columns=dataset['train'].column_names,
        fn_kwargs={'tokenizer': tokenizer, 'labels': labels}
    )

    dataset.set_format('torch')

    # Initialize model
    model = AutoModelForSequenceClassification.from_pretrained(opt['pretrained_model'],
                                                                problem_type=opt['problem_type'],
                                                                num_labels=len(labels),
                                                                id2label=id2label,
                                                                label2id=label2id)

    # Training arguments
    args = TrainingArguments(output_dir=make_current_datetime_dir(opt['output_dir']),
                             eval_strategy=opt['eval_strategy'],
                             save_strategy=opt['save_strategy'],
                             learning_rate=opt['learning_rate'],
                             per_device_train_batch_size=opt['per_device_train_batch_size'],
                             per_device_eval_batch_size=opt['per_device_eval_batch_size'],
                             num_train_epochs=opt['num_train_epochs'],
                             weight_decay=opt['weight_decay'],
                             load_best_model_at_end=opt['load_best_model_at_end'],
                             metric_for_best_model=opt['metric_for_best_model'],
                             seed=opt['seed'],
                             dataloader_num_workers=opt['dataloader_num_workers'],
                             no_cuda=opt['no_cuda']
                             )

    # Trainer setup
    trainer = Trainer(args=args,
                      model=model,
                      tokenizer=tokenizer,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset['val'],
                      compute_metrics=compute_metrics
                      )

    # Training the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    f1 = eval_results.get('eval_f1', 0)

    return f1  # Return the F1 score for Optuna to minimize/maximize


# Main code to start the Optuna optimization process
if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')  # We want to maximize the F1 score
    study.optimize(objective, n_trials=10)  # Number of trials to run

    # Print the best trial result
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best F1 score: {study.best_value}")
