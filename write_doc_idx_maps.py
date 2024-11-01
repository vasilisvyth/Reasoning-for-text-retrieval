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
r"""Script to write files for mapping documents to IDs.

Writes two output files:
* TSV file with document titles and ids.
* TSV file with document text and ids.

Convert to the indexed format used by the t5x_retrieval library.
"""

from absl import app
from absl import flags
import pickle
from quest.common import document_utils
from quest.common import tsv_utils

FLAGS = flags.FLAGS

flags.DEFINE_string("documents", "quest_data\\documents.jsonl", "Filepath of jsonl documents file.")

flags.DEFINE_string("doc_title_map", "quest_data\\doc_title_map.tsv",
                    "Filepath to write doc titles and ids.")

flags.DEFINE_string("doc_text_map", "quest_data\\doc_text_map.tsv", "Filepath to write doc text and ids.")


def main(unused_argv):
  documents = document_utils.read_documents(FLAGS.documents)
  doc_title_ids = []
  doc_text_ids = []
  for idx, document in enumerate(documents):
    doc_title_ids.append((idx, document.title))
    # Prepending document title to document text following previous work
    doc_text_ids.append(
        (idx, document.title + " " + document.text.replace("\n", " ")))

  tsv_utils.write_tsv(doc_title_ids, FLAGS.doc_title_map)
  tsv_utils.write_tsv(doc_text_ids, FLAGS.doc_text_map)

  # Serialize the data to a binary file using pickle.dump
  with open('quest_data\\doc_text_list.pickle', 'wb') as f:
      pickle.dump(doc_text_ids, f)

if __name__ == "__main__":
  app.run(main)
