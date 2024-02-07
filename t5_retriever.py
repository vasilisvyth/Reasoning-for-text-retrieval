from quest.common import example_utils
from quest.common import tsv_utils
import argparse
import os
import torch
from transformers import AutoTokenizer, TrainingArguments, Trainer, EarlyStoppingCallback, DataCollatorWithPadding
from transformers.optimization import AdafactorSchedule
from data.pair_dataset import PairDataset
from data.pair_collator import BiEncoderPairCollator
from pathlib import Path
from bi_encoder import DenseBiEncoder
from transformers import T5EncoderModel,  Adafactor#, AdafactorSchedule
from data.prepare_dataset import build_positive_pairs, read_docs, read_queries, remove_complex_queries, split
from data.evaluate_dataset import  EvaluateDocsDataset, EvaluateQueryDataset
from data.evaluate_collator import EvaluateCollator
from hf_trainer import HFTrainer
from seeds import set_seed
import uuid
import wandb
from torch import optim


def print_args(args):
    # Print the values using a for loop
    print("Argument values:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

def main(args):
    # args.pretrained = 'google/t5-v1_1-base'
    args.use_complex_queries = True
    # args.do_only_eval  = True
    print_args(args)
    set_seed(args.seed)
    
    # load model and tokenizer
    model = DenseBiEncoder(args.pretrained, args.scale_logits, args.right_loss)
    # model = DenseBiEncoder.from_pretrained(args.pretrained, args.scale_logits, args.right_loss)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    
    # Load training and validation examples
    examples_train_aug = example_utils.read_examples(os.path.join(args.data_dir,'train_aug.jsonl'))
    examples_val = example_utils.read_examples(os.path.join(args.data_dir,'val.jsonl'))
    examples_test = example_utils.read_examples(os.path.join(args.data_dir,'test.jsonl'))
    
    # Load training data related files
    train_dict_query_ids_queries, train_query_ids_doc_ids = read_queries(os.path.join(args.data_dir, 'train_query_ids_queries.tsv'), 
                                                                         os.path.join(args.data_dir,'train_query_ids_doc_ids.tsv'))

    # load docs
    doc_text_map, doc_title_map = read_docs(os.path.join(args.data_dir,'doc_text_list.pickle'), os.path.join(args.data_dir,'doc_title_map.tsv'))

    # Remove queries with metadata!= '_' from the dictionary
    if not args.use_complex_queries:
        print(f'Training set Total queries before removing complex queries: {len(train_dict_query_ids_queries)}')
        remove_complex_queries(examples_train_aug, train_dict_query_ids_queries)
        print(f'Training set Total queries after removing complex queries: {len(train_dict_query_ids_queries)}')
        # didn't remove from train_query_ids_doc_ids
    if args.split:
        train_dict_query_ids_queries, train_query_ids_doc_ids, examples_train_aug,\
        val_dict_query_ids_queries_extra, _, examples_val_extra \
        = split(train_dict_query_ids_queries, train_query_ids_doc_ids, examples_train_aug) 

    # build positive pairs for training
    train_query_ids, train_doc_ids, train_queries, train_docs = build_positive_pairs(train_dict_query_ids_queries, train_query_ids_doc_ids, doc_text_map)

    # Load validation and test data related files
    val_dict_query_ids_queries, _ = read_queries(os.path.join(args.data_dir,'val_query_ids_queries.tsv'), 
                                                 os.path.join(args.data_dir,'val_query_ids_doc_ids.tsv'))
    
    test_dict_query_ids_queries, _ = read_queries(os.path.join(args.data_dir,'test_query_ids_queries.tsv'), 
                                                 os.path.join(args.data_dir,'test_query_ids_doc_ids.tsv'))
    if args.split:
        val_dict_query_ids_queries.update(val_dict_query_ids_queries_extra)
        examples_val.extend(examples_val_extra)


    # if we don't want to use complex queries then remove them
    if not args.use_complex_queries:
        print(f'Val set Total queries before removing complex queries: {len(val_dict_query_ids_queries)}')
        remove_complex_queries(examples_val, val_dict_query_ids_queries)
        print(f'Val set Total queries after removing complex queries: {len(val_dict_query_ids_queries)}')

    if 'bge' in args.tokenizer:
        train_queries = ['Represent this sentence for searching relevant passages: '+query for query in train_queries]
    

    # create dataset for training
    train_pair_dataset = PairDataset(train_query_ids, train_doc_ids, train_queries, train_docs,examples_train_aug)

    # create dataset for validation
    eval_dataset = {}
    eval_dataset['queries']= EvaluateQueryDataset(examples_val, val_dict_query_ids_queries, tokenizer)
    eval_dataset['docs'] = EvaluateDocsDataset(doc_text_map, tokenizer)

    # create dataset for test
    test_dataset = {}
    test_dataset['queries']= EvaluateQueryDataset(examples_test, test_dict_query_ids_queries, tokenizer)
    test_dataset['docs'] = eval_dataset['docs']#EvaluateDocsDataset(doc_text_map, tokenizer)

    # Create collators for validation and training pairs
    train_collator  = BiEncoderPairCollator(
            tokenizer, query_max_length = 64, doc_max_length = 512
    )
    query_val_collator = DataCollatorWithPadding(max_length=64, tokenizer=tokenizer, padding='max_length')
    doc_val_collator = DataCollatorWithPadding(max_length=512, tokenizer=tokenizer, padding='max_length')
    dict_val_collator = {'queries':query_val_collator,'docs':doc_val_collator}

    random_identifier = str(uuid.uuid4())[:11]
    OUTPUT_DIR = Path(args.output_dir) / args.pretrained / "model" / random_identifier
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    #key d05e9001945806ce8dcba485dc234bfe7579ca55
    os.environ["WANDB_PROJECT"] = "quest" # name your W&B project 
    # os.environ["WANDB_LOG_MODEL"] = "checkpoint" # log all model checkpoints

    # wandb.login()
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=args.lr,
        # torch_compile=True,
        evaluation_strategy="steps",
        fp16=args.fp16,
        # fp16=True,
        warmup_steps=args.warmup_steps,
        metric_for_best_model=f'avg_rec@{args.k_for_eval}',
        load_best_model_at_end=True,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        max_steps=args.max_steps,
        save_steps=args.save_steps, #Number of updates steps before two checkpoint saves
        eval_steps=args.eval_steps, #Number of update steps between two evaluations
        save_total_limit=2, # saves best and another one
        # optim = 'adafactor',
        eval_delay = 10000,
        # lr_scheduler_type='linear',
        report_to="wandb",
        run_name=args.wb_run_name,  # name of the W&B run (optional)
        dataloader_num_workers = args.dataloader_num_workers,
        # auto_find_batch_size = True
        logging_steps=10  # how often to log to W&B
    )

    # scale_parameter, relative_step unsure have to compare with jax implementation
    scale_parameter=True
    relative_step=False
    warmup_init = False # warmup_init=True` requires `relative_step=True
    print(f'scale_parameter {scale_parameter} relative_step {relative_step} warmup_init {warmup_init}')
    if 't5' in args.pretrained:
        optimizer = Adafactor(model.parameters(), scale_parameter=scale_parameter, relative_step=relative_step, warmup_init=warmup_init, lr=args.lr)
    else:
        optimizer=optim.AdamW(model.parameters(),lr=args.lr, betas=(0.9,0.95))
    # lr_scheduler = AdafactorSchedule(optimizer)

    # trainer = Trainer(model, training_args, data_collator = pair_collator, train_dataset=train_pair_dataset,
    #                   eval_dataset=eval_dataset)#, callbacks=[],optimizers=(optimizer, lr_scheduler))  
    early_stopping = EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.05)
    
    #     model(batch['query_ids'], batch['doc_ids'], batch['queries'], batch['docs'])
    trainer = HFTrainer(
        model,
        train_dataset=train_pair_dataset,
        data_collator=train_collator,
        args=training_args,
        eval_dataset=test_dataset,
        eval_collator=dict_val_collator,
        eval_k = args.k_for_eval,
        doc_title_map = doc_title_map,
        fp_16=args.fp16,
        callbacks=[early_stopping],
        #dataloader_num_workers 
        optimizers = (optimizer, None)
    )
    # train_dataloader = trainer.get_train_dataloader()
    # for batch in train_dataloader:
    #     a=1
    if args.do_train:
        trainer.train()
        trainer.save_model()
    
    if args.do_only_eval:
        print('-'*16)
        print('DO ONLY EVALUATION')
        trainer.evaluate(test_dataset)
        print('-'*16)
    #wandb.finish() # for colab only

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Training Neural IR models")

    parser.add_argument(
        "--pretrained",
        type=str,
        default="google/t5-v1_1-small", #google/t5-v1_1-small
        help="Pretrained checkpoint for the base model",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="google/t5-v1_1-small",
        help="Pretrained checkpoint for the base model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Output directory to store models and results after training",
    )

    parser.add_argument(
        "--train_batch_size", type=int, default=2, help="Training batch size"
    ) # multipliers of 64 recommended for A100
    parser.add_argument(
        "--eval_batch_size", type=int, default=2, help="Evaluation batch size"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed to use"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=0, help="warmup steps"
    )
    parser.add_argument(
        "--max_steps", type=int, default=2, help="Number of training steps"
    )
    parser.add_argument(
        "--wb_run_name", type=str, default='debug', help='name of weight and bias run',
    )
    parser.add_argument(
        "--scale_logits", action='store_true', help='whether to scale the logits with 100',
    ) # default is false
    parser.add_argument(
        "--right_loss", action='store_true', help='whether to calculate the right',
    ) # default is false
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Learning rate for training"
    ) 
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=0, help="number of workers"
    ) 
    parser.add_argument(
        "--use_complex_queries", action='store_true', help='whether to calculate the right'
    ) # default false
    parser.add_argument(
        "--split", action='store_true', help='whether to consider extra 250 instances from training as validation'
    ) # default false
    parser.add_argument(
        "--fp16", action='store_true', help='whether to use fp16 '
    ) # default false
    parser.add_argument(
        "--do_train", action='store_true', help='whether to do training '
    ) # default false
    parser.add_argument(
        "--do_only_eval", action='store_true', help='whether to do evaluation ONLY'
    ) # default false
    parser.add_argument(
        "--k_for_eval", type=int, default=20, help="top k used to select the best model using a Eval_metric@k"
    )
    parser.add_argument(
        "--data_dir", type=str, default='quest_data', help="The data folder where you have the data"
    )
    parser.add_argument(
        "--save_steps", type=int, default=2, help="Number of updates steps before two checkpoint saves"
    )
    parser.add_argument(
        "--eval_steps", type=int, default=2, help="evaluate every"
    )

    # save_steps=100, #Number of updates steps before two checkpoint saves
    #     eval_steps=5,
    args = parser.parse_args()
    main(args)