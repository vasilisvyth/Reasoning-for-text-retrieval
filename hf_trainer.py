from transformers import Trainer
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
import torch
# import ir_measures
# from ir_measures import *
from analyze_retriever import calc_mrec_rec
import os
from transformers.utils import WEIGHTS_NAME
import logging
from quest.common.example_utils import Example
from typing import Dict

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HFTrainer(Trainer):
    def __init__(self, *args, eval_collator=None, eval_k, doc_title_map, fp_16,**kwargs):
        super().__init__(*args, **kwargs)
        self.eval_collator = eval_collator
        self.eval_k = eval_k
        self.doc_title_map = doc_title_map
        self.fp_16 = fp_16

    def get_eval_dataloader(self, eval_dataset=None, str_type = None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        data_collator = self.eval_collator[str_type]
        eval_sampler = self._get_eval_sampler(eval_dataset)
        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory
        )

    def evaluate(
        self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval",
    ):
        queries_data_loader = self.get_eval_dataloader(eval_dataset['queries'],'queries')
        docs_data_loader = self.get_eval_dataloader(eval_dataset['docs'],'docs')
        gold_examples = self.eval_dataset['queries'].gold_examples
        self.model.eval()
        logger.info("Running evaluation")

        device = next(self.model.parameters()).device
        print(f'device {device}')

        all_dids = []
        all_docs_scores = []
        i=0
        for batch in tqdm(
            docs_data_loader, desc="Encoding the docs", position=0, leave=True
        ):
   
            dids = batch.pop('ids')
            in_ids =  batch["inputs"]['input_ids'].to(device)
            att_mask = batch["inputs"]['attention_mask'].to(device)
            # if self.fp_16:
            with torch.cuda.amp.autocast(), torch.no_grad(): # scores are float32
                docs_scores = self.model.encode(input_ids =in_ids, attention_mask = att_mask).cpu()
            
            all_dids.extend(dids)
            all_docs_scores.append(docs_scores)
            

        all_docs_scores = torch.cat(all_docs_scores,dim=0)
        all_pred_examples = []

        for batch in tqdm(
            queries_data_loader, desc="Encoding the queries", position=0, leave=True
        ):
       
            qids = batch.pop('ids')
            in_ids =  batch["inputs"]['input_ids'].to(device)
            att_mask = batch["inputs"]['attention_mask'].to(device)
            with torch.cuda.amp.autocast(), torch.no_grad():
                queries_scores = self.model.encode(input_ids =in_ids, attention_mask = att_mask).cpu()
            
            similarities = torch.matmul(queries_scores, all_docs_scores.t())
            top_k_values, top_k_indices = torch.topk(similarities, self.eval_k, dim=1, sorted=True)

    
            # Creating examples and adding them to the list
            for i, qid in enumerate(qids):
                query_text = eval_dataset['queries'].dict_query_ids_queries[qid]
                doc_texts = [self.doc_title_map[all_dids[index]] for index in top_k_indices[i].tolist()]
                scores = top_k_values[i].tolist()
                # scores

                example = Example(query=query_text, docs=doc_texts, scores=scores)
                all_pred_examples.append(example)
            

        avg_recall_vals, avg_mrecall_vals = calc_mrec_rec(gold_examples, all_pred_examples)

        self.model.train()
  
        metrics = {metric_key_prefix + "_" + f'avg_rec@{self.eval_k}': avg_recall_vals[self.eval_k]}
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
        # if self.data_collator.tokenizer is not None:
           
        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, 'my_args.bin'))
    

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            # if is_torch_tpu_available():
            #     xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                # for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=self.eval_dataset,
                    ignore_keys=ignore_keys_for_eval
                )
                metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)