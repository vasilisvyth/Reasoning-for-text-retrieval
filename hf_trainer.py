from transformers import Trainer
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
import ir_measures
from ir_measures import *
import os
from transformers.utils import WEIGHTS_NAME
import logging

TRAINING_ARGS_NAME = "training_args.bin"
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HFTrainer(Trainer):
    def __init__(self, *args, eval_collator=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator

    # def get_eval_dataloader(self, eval_dataset=None):
    #     if eval_dataset is None:
    #         eval_dataset = self.eval_dataset
    #     data_collator = self.eval_collator
    #     eval_sampler = self._get_eval_sampler(eval_dataset)
    #     return DataLoader(
    #         eval_dataset,
    #         sampler=eval_sampler,
    #         batch_size=self.args.eval_batch_size,
    #         collate_fn=data_collator,
    #         drop_last=self.args.dataloader_drop_last,
    #         num_workers=self.args.dataloader_num_workers,
    #         pin_memory=self.args.dataloader_pin_memory,
    #     )

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval",
    ):
        queries_data_loader = self.get_eval_dataloader(eval_dataset['queries'])
        docs_data_loader = self.get_eval_dataloader(eval_dataset['docs'])

        self.model.eval()
        logger.info("Running evaluation")

        all_qids = []
        all_subqids = []
        all_dids = []
        all_scores = []
        for batch in tqdm(
            docs_data_loader, desc="Encoding the docs", position=0, leave=True
        ):
            qids = batch.pop("query_ids")
            dids = batch.pop("doc_ids")
            subqids = batch.pop("subq_ids")

            batch_queries = {k: v.to(self.args.device) for k, v in batch['queries'].items()}
            batch_docs = {k: v.to(self.args.device) for k, v in batch['docs'].items()}
            with torch.no_grad():
                scores = self.model.score_pairs(batch_queries, batch_docs)
            
            all_qids.extend(qids.tolist())
            all_dids.extend(dids.tolist())
            all_scores.extend(scores.tolist())
            all_subqids.extend(subqids.tolist())

        dict_scores = {}
        for qid, did, score, subqid in zip(all_qids, all_dids, all_scores, all_subqids):
            if qid not in dict_scores:
                dict_scores[qid] = {}
            if did not in dict_scores[qid]:
                dict_scores[qid][did] = {}
            dict_scores[qid][did][subqid] = score
        
        # alternative
        # append((doc, score))

        # # Sort the documents within each group based on scores
        # for query in grouped_data:
        #     grouped_data[query] = sorted(grouped_data[query], key=lambda x: x[1])

        self.model.train()
        # qrels = (
        #     eval_dataset.qrels if eval_dataset is not None else self.eval_dataset.qrels
        # )
        # metrics = ir_measures.calc_aggregate(
        #     [nDCG @ 10, MRR @ 10, R @ 1000], qrels, rerank_run
        # )
        metrics = {metric_key_prefix + "_" + str(k): v for k, v in metrics.items()}
        metrics["epoch"] = self.state.epoch
        self.log(metrics)
        return metrics

    def _load_best_model(self):
        logger.info(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric})."
        )
        best_model_path = os.path.join(self.state.best_model_checkpoint, WEIGHTS_NAME)
        state_dict = torch.load(best_model_path, map_location="cpu")
        self.model.model.load_state_dict(state_dict, False)

    def _save(self, output_dir: str = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")
        self.model.save_pretrained(output_dir, state_dict=state_dict)
        if self.data_collator.tokenizer is not None:
            self.data_collator.tokenizer.save_pretrained(output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))