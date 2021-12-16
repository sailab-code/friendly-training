import torch
import torch.nn as nn
import torch.utils.data
import os
import re
import math



class TextManager:
    """Class that models the basic functionalities for preparing text in order to process it by neural networks."""

    def __init__(self, text_data_paths=None, max_sequence_length=0):
        """Initialize the text manager.

        Args:
            text_data_paths: a list of paths to the locations in which data is stored.
            max_sequence_length: the max length of each sequence (sequence-based representation), or -1 for no limits.
        """

        # attributes
        self.text_data_paths = text_data_paths  # path(s) to the root of the data folder (it can be a list of paths)
        self.vocab = None  # vocabulary (dictionary that maps a word to its ID)
        self.vocab_idf = None  # inverse document frequencies of each term in the vocabulary (term-ID to IDF)
        self.max_sequence_length = max_sequence_length  # max length of a sequence (if 0: no limits)
        self.padding_idx = -1  # the index of the sequence-padding element
        self.oov_idx = -1  # the index of out-of-vocabulary words
        self.vocab_size = -1  # the size of the vocabulary (also considering the padding element and the OOV element)

        # simple argument check
        if text_data_paths is not None:
            for path in text_data_paths:
                if not os.path.exists(path) or os.path.isfile(path):
                    raise ValueError("Invalid data path: " + str(path))
        if max_sequence_length is None or max_sequence_length < 0:
            raise ValueError("Invalid maximum sequence length")

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

        # storing file names
        text_files = []
        for text_data_path in self.text_data_paths:
            folder_contents = os.listdir(text_data_path)
            files = [os.path.join(text_data_path, f) for f in folder_contents
                     if os.path.isfile(os.path.join(text_data_path, f)) and f.endswith(".txt")]
            text_files.extend(files)

        # counting
        word_to_counts = {}  # word -> number of times this word appears in the dataset
        word_to_docs = {}  # word -> number of document with such word
        max_words = 0
        for i in range(0, len(text_files)):

            # here we load the reviews
            with open(os.path.join(text_files[i]), 'r', encoding='utf-8') as f:
                review_text = f.read()
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
        N = len(text_files)

        for word, count in word_count_pairs:
            self.vocab[word] = i
            self.vocab_idf.append(math.log(N / word_to_docs[word]))  # we also create the IDF
            i += 1
            if i >= max_len:  # cutting
                break

        self.oov_idx = len(self.vocab)
        self.padding_idx = len(self.vocab) + 1
        self.vocab_size = len(self.vocab) + 2

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
        self.path = path  # path to the root of the data folder
        self.text_manager = text_manager  # text manager object
        self.data_type = data_type  # type of data that will be stored/created in this dataset.
        self.files = []  # list composed of the names of the dataset files
        self.reviews = []  # list composed of the reviews loaded from the dataset files, sequence of token IDs
        self.reviews_length = []  # list composed of the lengths (number of tokens) of each preloaded review
        self.labels = []  # sentiment polarities (0 = negative, 1 = positive)

        # simple argument check
        if path is None:
            raise ValueError("You must specify the dataset path!")
        if not os.path.exists(path) or os.path.isfile(path):
            raise ValueError("Invalid data path: " + str(path))
        if data_type is None or (data_type != 'sequences' and data_type != 'bow'):
            raise ValueError("Unknown data type")
        if text_manager is None or type(text_manager) is not TextManager:
            raise ValueError("Invalid text manager!")

        # if we ask for an 'empty_dataset', no further actions are taken, otherwise file names and labels are loaded
        if not empty_dataset:
            sub_folders = ['neg', 'pos']

            # storing file names and labels
            j = 0
            for sub_folder in sub_folders:

                # getting file names for each class
                class_folder = os.path.join(self.path, sub_folder)
                folder_contents = os.listdir(class_folder)
                files = [os.path.join(class_folder, f) for f in folder_contents
                         if os.path.isfile(os.path.join(class_folder, f)) and f.endswith(".txt")]

                # storing the file names in the dataset
                self.files.extend(files)

                # storing labels in the dataset (each file of the class inherit the labeling provided for such class)
                self.labels.extend([j] * len(files))
                j += 1

    def __len__(self):
        """The total number of examples in this dataset."""

        return len(self.files)

    def preload_reviews(self, lengths_only=False):
        """Load all the reviews into the memory of the machine (optional operation - it speedups computations)."""

        self.reviews = []  # clearing
        self.reviews_length = []  # clearing

        for i in range(0, len(self.files)):
            with open(os.path.join(self.files[i]), 'r', encoding='utf-8') as f:
                review_text = f.read()
                if self.data_type == 'sequences':
                    review, review_length = self.text_manager.create_sequence_of_word_ids(review_text)
                else:
                    review = self.text_manager.create_bag_of_words(review_text)
                    review_length = torch.tensor(self.text_manager.vocab_size, dtype=torch.long)  # dummy

                if not lengths_only:
                    self.reviews.append(review)
                self.reviews_length.append(review_length)

    def __getitem__(self, index):
        """Load and return the representation of the next review from disk.

        Args:
            index: the index of the element to be loaded.

        Returns:
            The review representation, the reverted review (if sequences), its length (if sequences), the label.
        """

        # loading review
        if len(self.reviews) == 0:
            with open(os.path.join(self.files[index]), 'r', encoding='utf-8') as f:

                # reading review text
                review_text = f.read()

                # converting text into the selected representation
                if self.data_type == 'sequences':
                    review, review_length = self.text_manager.create_sequence_of_word_ids(review_text)  # sequence
                    reversed_review = torch.flip(review, dims=[0])  # reversing the sequence
                else:
                    review = self.text_manager.create_bag_of_words(review_text)  # bag-of-words
                    reversed_review = review  # dummy
                    review_length = torch.tensor(self.text_manager.vocab_size, dtype=torch.long)  # dummy
        else:
            review = self.reviews[index]  # if reviews were pre-loaded, then we simply pick them up
            review_length = self.reviews_length[index]

            if self.data_type == 'sequences':
                reversed_review = torch.flip(review, dims=[0])  # reversing the sequence
            else:
                reversed_review = review  # dummy

        # getting the label
        label = torch.tensor(self.labels[index], dtype=torch.float32)

        return review, reversed_review, review_length, label

    def pack_minibatch(self, data):
        """Pack a list of examples into tensors, commonly done when packing mini-batches.

        Args:
            data: a list of tuples, where each tuple is (review, reversed_review, review_length, label)

        Returns:
            Tensor of the (padded) reviews, tensor of the reversed reviews, label tensor, review length tensor.
        """
        reviews, reversed_reviews, review_lengths, labels = zip(*data)

        # pad_sequence will pad and pack the reviews such that we get a 2D tensor: sequence_lengths x batch_size
        if self.data_type == 'sequences':
            reviews = nn.utils.rnn.pad_sequence(reviews, padding_value=self.text_manager.padding_idx)
            reversed_reviews = nn.utils.rnn.pad_sequence(reversed_reviews, padding_value=self.text_manager.padding_idx)
        else:
            reviews = torch.stack(reviews, dim=0)
            reversed_reviews = reviews  # dummy
        review_lengths = torch.stack(review_lengths, dim=0)
        labels = torch.stack(labels, dim=0)

        return reviews, reversed_reviews, review_lengths, labels





# entry point
if __name__ == "__main__":
        max_sequence_length = 0
        output_folder = "reviews"
        data_type = "bow"
        dataset_folder = os.path.join("reviews","aclImdb")
        max_vocabulary_size = 20000

        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # data paths (path of the positive reviews, path of the negative reviews)
        _pos_text = os.path.join(dataset_folder, 'train', 'pos')
        _neg_text = os.path.join(dataset_folder, 'train', 'neg')

        # creating and saving a valid text manager
        _text_manager = TextManager([_pos_text, _neg_text], max_sequence_length=max_sequence_length)
        _text_manager.create_vocabulary(max_vocabulary_size)  # creating the vocabulary
        _text_manager.save(os.path.join(output_folder, 'vocabulary.dat'))  # saving to disk


        # preparing dataset
        _data_set = Dataset(os.path.join(dataset_folder, 'train'), _text_manager, data_type)
        _test_set = Dataset(os.path.join(dataset_folder, 'test'), _text_manager, data_type)

        # you can decide if you want to load the whole dataset in memory or not
        print("Preloading data...")
        _data_set.preload_reviews()
        _test_set.preload_reviews()

        print("Saving data to disk...")
        torch.save(torch.vstack(_data_set.reviews), os.path.join(output_folder,'reviews_tfidf_train.pt'))
        torch.save(torch.vstack(_test_set.reviews), os.path.join(output_folder,'reviews_tfidf_test.pt'))
        torch.save(torch.tensor(_data_set.labels), os.path.join(output_folder,'reviews_labels_train.pt'))
        torch.save(torch.tensor(_test_set.labels), os.path.join(output_folder,'reviews_labels_test.pt'))

        print("Save done!")
