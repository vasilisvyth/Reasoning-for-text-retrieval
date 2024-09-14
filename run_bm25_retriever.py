
"""Generate predicted candidates with BM25."""

import random

from absl import app
from absl import flags
import numpy as np
from quest.bm25 import bm25_retriever
from quest.common import document_utils
from quest.common import example_utils
import pickle


FLAGS = flags.FLAGS

flags.DEFINE_string("examples", "quest_data\\test.jsonl", "Path to examples jsonl file.")

flags.DEFINE_string("docs", "quest_data\\documents.jsonl", "Path to document corpus jsonl file.")

flags.DEFINE_string("output", "bm25.out", "Path to write predictions jsonl file.")

flags.DEFINE_integer("sample", 0, "Number of examples to sample if > 0.")

flags.DEFINE_integer("topk", 1000, "Number of documents to retrieve.")


def main(unused_argv):
  """
  First, they read the queries with the candidates documents titles.
  Then, they read the whole documents and calculate tf, idf etc. on them
  """
  examples = example_utils.read_examples(FLAGS.examples)


  with open('dum_bm25_obj.pickle', 'rb') as f:
    retriever = pickle.load(f)
  assert (retriever.bm25.idf == retriever.bm25.idf)
  predictions = []
  for idx, example in enumerate(examples):
    print("Processing example %s." % idx)
    docs_scores = retriever.get_docs_and_scores(example.query, topk=FLAGS.topk)
    docs = [doc for doc, _ in docs_scores]
    scores = [score for _, score in docs_scores]
    predictions.append(
        example_utils.Example(
            original_query=example.original_query,
            query=example.query,
            docs=docs,
            # scores=scores,
            metadata=example.metadata
        )
    )

  example_utils.write_examples(FLAGS.output, predictions)


if __name__ == "__main__":
  app.run(main)
