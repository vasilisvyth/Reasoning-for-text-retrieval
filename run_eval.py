"""Computes end-to-end evaluation metrics."""

from absl import app
from absl import flags

from quest.common import example_utils
from quest.eval import eval_utils


def calc_f1_pr_rec(gold_examples, pred_examples):
  #! todo return for training
  # List of values for average precision, recall, and f1.
  p_vals = []
  r_vals = []
  f1_vals = []

  query_to_pred_example = {ex.query: ex for ex in pred_examples}
  for gold_example in gold_examples:
    if not gold_example.docs:
      raise ValueError("Example has 0 docs.")

    pred_example = query_to_pred_example[gold_example.query]

    predicted_docs = set(pred_example.docs)
    gold_docs = set(gold_example.docs)
    tp = len(gold_docs.intersection(predicted_docs))
    fp = len(predicted_docs.difference(gold_docs))
    fn = len(gold_docs.difference(predicted_docs))
    if tp:
      precision = tp / (tp + fp)
      recall = tp / (tp + fn)
      f1 = 2 * precision * recall / (precision + recall)
    else:
      precision = 0.0
      recall = 0.0
      f1 = 0.0

    p_vals.append(precision)
    r_vals.append(recall)
    f1_vals.append(f1)

  print("Avg. Precision")
  avg_prec =eval_utils.print_avg(gold_examples, p_vals)
  template_avg_prec = eval_utils.print_avg_by_template(gold_examples, p_vals)
  print("Avg. Recall")
  avg_rec = eval_utils.print_avg(gold_examples, r_vals)
  template_avg_rec = eval_utils.print_avg_by_template(gold_examples, r_vals)
  print("Avg. F1")
  avg_f1 = eval_utils.print_avg(gold_examples, f1_vals)
  template_avg_f1 = eval_utils.print_avg_by_template(gold_examples, f1_vals)
  return avg_prec, avg_rec, avg_f1

def main(unused_argv):
  gold_examples = example_utils.read_examples(FLAGS.gold)
  pred_examples = example_utils.read_examples(FLAGS.pred)

  calc_f1_pr_rec(gold_examples, pred_examples)


if __name__ == "__main__":
  app.run(main)
