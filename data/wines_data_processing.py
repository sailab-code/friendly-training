import torch
import torch.utils.data
import os
import re
import math
import csv

from sentence_transformers import SentenceTransformer
sentence_encoder = SentenceTransformer('paraphrase-distilroberta-base-v1')

class TextManager:
    """Class that models the basic functionalities for preparing text in order to process it by neural networks."""

    def __init__(self, text_data_path=None, max_sequence_length=0):
        """Initialize the text manager.

        Args:
            text_data_paths: a list of paths to the locations in which data is stored.
            max_sequence_length: the max length of each sequence (sequence-based representation), or -1 for no limits.
        """

        # attributes
        self.text_data_path = text_data_path  # path(s) to the root of the data folder
        self.vocab = None  # vocabulary (dictionary that maps a word to its ID)
        self.vocab_idf = None  # inverse document frequencies of each term in the vocabulary (term-ID to IDF)
        self.max_sequence_length = max_sequence_length  # max length of a sequence (if 0: no limits)
        self.padding_idx = -1  # the index of the sequence-padding element
        self.oov_idx = -1  # the index of out-of-vocabulary words
        self.vocab_size = -1  # the size of the vocabulary (also considering the padding element and the OOV element)

        # simple argument check
        if text_data_path is not None:
            if not os.path.exists(text_data_path):
                raise ValueError("Invalid data path: " + str(path))
        if max_sequence_length is None or max_sequence_length < 0:
            raise ValueError("Invalid maximum sequence length")
        self.read_documents()

    def read_documents(self):
        self.documents = []

        with open(self.text_data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                # here we load the reviews
                if line_count == 0:
                    line_count += 1
                    continue
                else:
                    review_text = row[2]
                    line_count += 1
                    self.documents.append(review_text)
                
    def save(self, file):
        """Save the vocabulary (and related information) to file."""

        with open(file, 'w+', encoding='utf-8') as f:
            for word, idx in self.vocab.items():
                f.write("{} {} {}\n".format(idx, word, self.vocab_idf[idx]))

    def load(self, file):
        """Load the vocabulary (and related information) from file."""

        self.vocab = {}
        self.vocab_idf = []

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line_fields = line.split()
                self.vocab[line_fields[1]] = int(line_fields[0])
                self.vocab_idf.append(float(line_fields[2]))

        self.oov_idx = len(self.vocab)
        self.padding_idx = len(self.vocab) + 1
        self.vocab_size = len(self.vocab) + 2

    @staticmethod
    def __word_tokenization(text):
        """Brutal-and-insanely-simple tokenizer that splits text into words."""

        return re.findall("[\\w]+", text.lower())

    def create_vocabulary(self, max_len=20000):
        """Create the vocabulary of words (keeping the most frequent ones), assigning a unique index to each word."""


        # counting
        word_to_counts = {}  # word -> number of times this word appears in the dataset
        word_to_docs = {}  # word -> number of document with such word
        max_words = 0
        for review_text in self.documents:
                review_text_tokenized = TextManager.__word_tokenization(review_text)
                max_words = max(max_words, len(review_text_tokenized))

                considered_words = {}

                for word in review_text_tokenized:
                    if word in word_to_counts:
                        word_to_counts[word] += 1

                        if word not in considered_words:
                            considered_words[word] = True
                            word_to_docs[word] += 1
                    else:
                        word_to_counts[word] = 1
                        word_to_docs[word] = 1
                        considered_words[word] = True

        # sorting words in order to put the most frequent ones on top
        word_count_pairs = sorted(word_to_counts.items(), key=lambda x: x[1], reverse=True)

        # creating the real vocabulary
        self.vocab = {}
        self.vocab_idf = []
        i = 0
        N = len(self.documents)

        for word, count in word_count_pairs:
            self.vocab[word] = i
            self.vocab_idf.append(math.log(N / word_to_docs[word]))  # we also create the IDF
            i += 1
            if i >= max_len:  # cutting
                break

        self.oov_idx = len(self.vocab)
        self.padding_idx = len(self.vocab) + 1
        self.vocab_size = len(self.vocab) + 2

    def load_embedding_matrix(self, matrix_file):
        """Load word embeddings for the current vocabulary of words from a text file.

        Args:
            matrix_file: a text file with a word embedding per-row (numerical features are separated by spaces;
                each row starts with the word name).

        Returns:
            Embedding matrix for the current vocabulary (missing embeddings are randomly set), one embedding per row.
        """

        if self.vocab is None:
            raise ValueError("Build a vocabulary before looking for word embeddings!")

        # guessing the size of each embedding vector
        embedding_size = -1
        with open(matrix_file, 'r') as file:
            for line in file:
                line_fields = line.split()
                embedding_size = len(line_fields) - 1
                break

        # filling the embedding matrix for the current vocabulary words
        embedding_matrix = torch.randn((self.vocab_size, embedding_size), dtype=torch.float32)  # random init

        with open(matrix_file, 'r', encoding='utf-8') as file:
            for line in file:
                line_fields = line.split()
                word = line_fields[0].lower()  # since our vocabulary is lowercase, we enforce lowercase here too

                if word in self.vocab:  # if the word is in our vocabulary, we copy the loaded embedding vector
                    word_id = self.vocab[word]
                    embedding_matrix[word_id, :] = torch.tensor([float(i) for i in line_fields[1:]],
                                                                dtype=torch.float32)

        # embedding of the padding symbol (setting it to zeros, even if this embedding will be actually discarded)
        embedding_matrix[self.padding_idx, :] = 0. * embedding_matrix[self.padding_idx, :]
        return embedding_matrix

    def create_sequence_of_word_ids(self, text):
        """Convert plain text into a sequence of word IDs, accordingly to the dataset vocabulary."""

        if self.vocab is None:
            raise ValueError("Build a vocabulary first!")

        # splitting text into words
        text_tokenized = TextManager.__word_tokenization(text)

        # converting words to word IDs
        sequence_of_token_ids = []
        i = 0
        for word in text_tokenized:
            sequence_of_token_ids.append(self.vocab[word] if word in self.vocab else self.oov_idx)
            i += 1
            if 0 < self.max_sequence_length <= i:  # cutting
                break

        sequence_length = len(sequence_of_token_ids)
        return torch.tensor(sequence_of_token_ids, dtype=torch.long), torch.tensor(sequence_length, dtype=torch.long)

    def create_bag_of_words(self, text):
        """Convert plain text into a bag-of-words (TF/IDF), accordingly to the dataset vocabulary."""

        if self.vocab is None:
            raise ValueError("Build a vocabulary first!")

        # splitting text into words
        text_tokenized = TextManager.__word_tokenization(text)

        # counting how many times each word appears in this text
        counts = {}  # word ID -> number of times it appears in this text
        for word in text_tokenized:
            if word in self.vocab:
                token_id = self.vocab[word]
                if token_id in counts:
                    counts[token_id] += 1
                else:
                    counts[token_id] = 1
            else:
                token_id = self.oov_idx
                if token_id in counts:
                    counts[token_id] += 1
                else:
                    counts[token_id] = 1

        # creating the TD/IDF-based bag-of-words
        bow = torch.zeros(self.vocab_size, dtype=torch.float32)  # init all elements to zero
        for word in text_tokenized:
            if word in self.vocab:
                token_id = self.vocab[word]
                tf = 1. + math.log(counts[token_id])

                bow[token_id] = tf * self.vocab_idf[token_id]
            else:
                token_id = self.oov_idx
                tf = 1. + math.log(counts[token_id])

                bow[token_id] = tf * 0.05  # we are using a generic (low) IDF for OOV words (they are common)

        return bow


class Dataset(torch.utils.data.Dataset):
    """Class that models a generic dataset composed of review-related files and targets (sentiment polarities)."""

    def __init__(self, path, text_manager, data_type='sequences', empty_dataset=False):
        """Create a dataset.

        Args:
            path: string with the path to the dataset root.
            text_manager: a reference to a ready-to-use TextManager object that will handle this dataset.
            data_type: type of representation of the data, in {'bow', 'sequences'}.
            empty_dataset (optional): boolean flag that indicates whether the dataset should be empty.
        """

        # dataset attributes
        self.path = path  # path to the data file
        self.text_manager = text_manager  # text manager object
        self.data_type = data_type  # type of data that will be stored/created in this dataset.
        self.files = []  # list composed of the names of the dataset files
        self.reviews = []  # list composed of the reviews loaded from the dataset files, sequence of token IDs
        self.reviews_length = []  # list composed of the lengths (number of tokens) of each preloaded review
        self.labels = []  # sentiment polarities (0 = negative, 1 = positive)

        # simple argument check
        if path is None:
            raise ValueError("You must specify the dataset path!")
        if not os.path.isfile(path):
            raise ValueError("Invalid data path: " + str(path))
        if data_type is None or (data_type != 'sequences' and data_type != 'bow' and data_type != 'embeddings'):
            raise ValueError("Unknown data type")
        if text_manager is None or type(text_manager) is not TextManager:
            raise ValueError("Invalid text manager!")

        # if we ask for an 'empty_dataset', no further actions are taken, otherwise file names and labels are loaded
        if not empty_dataset:
            self.read_dataset()

    def read_dataset(self):
        with open(self.path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    review_text = row[2]
                    review_points = row[4]
                    if self.data_type == 'sequences':
                        review, review_length = self.text_manager.create_sequence_of_word_ids(review_text)
                    elif self.data_type == 'bow':
                        review = self.text_manager.create_bag_of_words(review_text)
                    elif self.data_type == 'embeddings':
                        review = torch.tensor(sentence_encoder.encode([review_text])[0])
                    self.reviews.append(review)
                    self.labels.append(int(int(review_points)>=90))
                    line_count += 1

    def __len__(self):
        """The total number of examples in this dataset."""
        return len(self.reviews)



# entry point
if __name__ == "__main__":
    max_sequence_length = 0
    output_folder = "wine"
    data_type = "embeddings"
    max_vocabulary_size = 20000
    outname = "winedr"

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # data paths (path of the positive reviews, path of the negative reviews)
    _csv = 'winemag-data-130k-v2.csv'

    # creating and saving a valid text manager
    _text_manager = TextManager(_csv, max_sequence_length=max_sequence_length)
    _text_manager.create_vocabulary(max_vocabulary_size)  # creating the vocabulary
    _text_manager.save(os.path.join(output_folder,'vocabulary.dat'))  # saving to disk

    # preparing dataset
    print("Preprocessing text (it may take some minutes)...")
    _data_set = Dataset(_csv, _text_manager, data_type)

    _train_data_set = _data_set.reviews[:100000]
    _test_data_set = _data_set.reviews[100000:]

    _train_labels = _data_set.labels[:100000]
    _test_labels = _data_set.labels[100000:]

    print("Saving data to disk...")
    torch.save(torch.vstack(_train_data_set), os.path.join(output_folder,outname+'_input_train.pt'))
    torch.save(torch.vstack(_test_data_set), os.path.join(output_folder,outname+'_input_test.pt'))
    torch.save(torch.tensor(_train_labels), os.path.join(output_folder,outname+'_labels_train.pt'))
    torch.save(torch.tensor(_test_labels), os.path.join(output_folder,outname+'_labels_test.pt'))

    print("Save done!")


