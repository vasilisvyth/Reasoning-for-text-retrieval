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
"""Utilities for reading and writing jsonl files."""

import json
from tensorflow.io import gfile


def read(filepath, limit=None, verbose=False):
  """Read jsonl file to a List of Dicts.
  ADDED BY ME:
    verbose: An optional boolean parameter that controls whether to print progress messages (default is False).

    Each examples file contains newline-separated json dictionaries with the following fields:

    query - Paraphrased query written by annotators.
    docs - List of relevant document titles.
    original_query - The original query which was paraphrased. Atomic queries are enclosed by <mark></mark>. Augmented queries do not have this field populated.
    scores - This field is not populated and only used when producing predictions to enable sharing the same data structure.
    metadata - A dictionary with the following fields:
        template - The template used to create the query.
        domain - The domain to which the query belongs.
        fluency - List of fluency ratings for the query.
        meaning - List of ratings for whether the paraphrased query meaning is the same as the original query.
        naturalness - List of naturalness ratings for the query.
        relevance_ratings - Dictionary mapping document titles to relevance ratings for the document.
        evidence_ratings - Dictionary mapping document titles to evidence ratings for the document.
        attributions - Dictionary mapping a document title to its attributions attributions are a list of dictionaries mapping a query substring to a document substring.

  """
  data = []
  templates = []
  with gfile.GFile(filepath, "r") as jsonl_file:
    for idx, line in enumerate(jsonl_file):
      if limit is not None and idx >= limit:
        break
      if verbose and idx % 100 == 0:
        # Print the index every 100 lines.
        print("Processing line %s." % idx)
      try:
        dict = json.loads(line)
        templates.append(dict['metadata']['template'])
        data.append(dict)
      except json.JSONDecodeError as e:
        print("Failed to parse line: `%s`" % line)
        raise e
  print("Loaded %s lines from %s." % (len(data), filepath))
  return data


def write(filepath, rows):
  """Write a List of Dicts to jsonl file."""
  with gfile.GFile(filepath, "w") as jsonl_file:
    for row in rows:
      line = "%s\n" % json.dumps(row)
      jsonl_file.write(line)
  print("Wrote %s lines to %s." % (len(rows), filepath))
