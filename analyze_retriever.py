"""Analyze retriever predictions."""

from absl import app
from absl import flags

from quest.common import example_utils
from quest.eval import eval_utils



offset = ''
limit = ''
verbose = False

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
  all_recall_per_template = {}
  for k in K_VALS:
    print("MRecall@%s" % k)

    mrecall_avg_all  = eval_utils.print_avg(gold_examples, mrecall_vals[k])
    mrecall_avg_by_template  = eval_utils.print_avg_by_template(gold_examples, mrecall_vals[k])
    print("Avg. Recall@%s" % k)
    recall_avg_all  = eval_utils.print_avg(gold_examples, recall_vals[k])
    recall_avg_by_template  = eval_utils.print_avg_by_template(gold_examples, recall_vals[k])

    recall_avg_by_template = {f'R@{str(k)}:{key}': value for key, value in recall_avg_by_template.items()}
    all_recall_per_template.update(recall_avg_by_template)
    
    avg_recall_vals[k] = recall_avg_all['all']
    avg_mrecall_vals[k] = mrecall_avg_all['all'] # I swapped order
    
  return avg_recall_vals, avg_mrecall_vals, all_recall_per_template


def main(unused_argv):
  gold_examples = example_utils.read_examples('quest_data\\test.jsonl')
  pred_examples = example_utils.read_examples('bm25.out')
  calc_mrec_rec(gold_examples, pred_examples)
  


if __name__ == "__main__":
  app.run(main)
