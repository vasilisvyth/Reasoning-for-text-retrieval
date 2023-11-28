# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Analyze retriever predictions."""

from absl import app
from absl import flags

from quest.common import example_utils
from quest.eval import eval_utils


# FLAGS = flags.FLAGS

# flags.DEFINE_string("gold", "", "Examples jsonl file with gold document set.")

# flags.DEFINE_string("pred", "", "Examples jsonl file with predicted documents.")

offset = ''
limit = ''
verbose = False
# flags.DEFINE_integer("offset", 0, "Start index for examples to process.")

# flags.DEFINE_integer("limit", 0, "End index for examples to process if >0.")

# flags.DEFINE_bool("verbose", False, "Whether to print out verbose debugging.")

# Values of k to use to compute MRecall@k.
K_VALS = [20, 50, 100, 1000]

def calc_mrec_rec(gold_examples, pred_examples):
  #! todo return for training
  num_examples = 0
  # Dictionary mapping mrecall k value to number of examples where predicted
  # set is superset of gold set.
  mrecall_vals = {k: [] for k in K_VALS} # for every k keep a list where every element in the list represents the score of the respective example
  # List of recall for each example.
  recall_vals = {k: [] for k in K_VALS}

  query_to_pred_example = {ex.query: ex for ex in pred_examples}
  for idx, gold_example in enumerate(gold_examples):
    if offset and idx < offset:
      continue
    if limit and idx >= limit:
      break

    if not gold_example.docs:
      raise ValueError("Example has 0 docs.")

    
    if gold_example.query in query_to_pred_example:

      if verbose:
        print("\n\nProcessing example %s: `%s`" % (idx, gold_example))
        num_examples += 1

      pred_example = query_to_pred_example[gold_example.query]

      for k in K_VALS:
        if verbose:
          print("Evaluating MRecall@%s" % k)
        predicted_docs = set(pred_example.docs[:k])
        gold_docs = set(gold_example.docs)
        if gold_docs.issubset(predicted_docs):
          if verbose:
            print("Contains all docs!")
          mrecall_vals[k].append(1.0)
        else:
          mrecall_vals[k].append(0.0)

        # Compute recall.
        covered_docs = gold_docs.intersection(predicted_docs)
        recall = float(len(covered_docs)) / len(gold_docs)
        recall_vals[k].append(recall)

        # Print debugging.
        extra_docs = predicted_docs.difference(gold_docs)
        missing_docs = gold_docs.difference(predicted_docs)
        if verbose:
          print("Extra docs: %s" % extra_docs)
          print("Missing docs: %s" % missing_docs)
    else:
      print('not in dict')

  print("num_examples: %s" % num_examples)

  avg_mrecall_vals = {k:0  for k in K_VALS}
  avg_recall_vals = {k:0  for k in K_VALS}
  for k in K_VALS:
    print("MRecall@%s" % k)

    mrecall_avg_all  = eval_utils.print_avg(gold_examples, mrecall_vals[k])
    mrecall_avg_by_template  = eval_utils.print_avg_by_template(gold_examples, mrecall_vals[k])
    print("Avg. Recall@%s" % k)
    recall_avg_all  = eval_utils.print_avg(gold_examples, recall_vals[k])
    recall_avg_by_template  = eval_utils.print_avg_by_template(gold_examples, recall_vals[k])
    # print('!'*17)
    # print('returned')
    # print('recall_avg_all', recall_avg_all)
    # print('mrecall_avg_all', mrecall_avg_all)
    avg_recall_vals[k] = recall_avg_all['all']
    avg_mrecall_vals[k] = mrecall_avg_all['all'] # I swapped order
    
  return avg_recall_vals, avg_mrecall_vals


def main(unused_argv):
  a=1
  # path_doc_text_list = os.path.join('quest_data','doc_text_list.pickle')
  # path_doc_title_map = os.path.join('quest_data','doc_title_map.tsv')
  # _, doc_title_map = read_docs(path_doc_text_list, path_doc_title_map)

  # path_val_query_ids_queries = os.path.join('quest_data','val_query_ids_queries.tsv')
  # path_val_query_ids_doc_ids = os.path.join('quest_data','val_query_ids_doc_ids.tsv')

  # val_dict_query_ids_queries, _ = read_queries(path_val_query_ids_queries, path_val_query_ids_doc_ids)
  # gold_path = os.path.join('quest_data','val.jsonl')
  # gold_examples = example_utils.read_examples(gold_path)
  # pred_examples = []
  # num_docs = len(doc_title_map)
  # for ex in gold_examples:
  #   query = ex.query
  #   docs = [doc_title_map[random.randint(0,num_docs)] for i in range(5)]
  #   pred_example = Example(query=query, docs=docs)
  #   pred_examples.append(pred_example)

  # # pred_examples = example_utils.read_examples(FLAGS.pred)
  # calc_mrec_rec(gold_examples, pred_examples)
  


if __name__ == "__main__":
  app.run(main)
