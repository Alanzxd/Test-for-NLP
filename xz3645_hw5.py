import argparse
import nltk
#python xz3645_hw5.py  WSJ_02-21.pos-chunk training.feature
#python xz3645_hw5.py  WSJ_23.pos test.feature
#java -cp .;maxent-3.0.0.jar;trove.jar MEtrain training.feature model.chunk
#java -cp .;maxent-3.0.0.jar;trove.jar MEtag test.feature model.chunk WSJ_23.chunk
stemmer = nltk.stem.SnowballStemmer('english')

class EnglishWord:
    def __init__(self, word, pos, capitalization_or_not, tag=None):
        self.word = word
        self.pos = pos
        self.capitalization_or_not = capitalization_or_not
        self.tag = tag

    def __str__(self):
        return f"{self.word}({self.pos}){' [CAP]' if self.capitalization_or_not else ''} - {self.tag}"

def parse(input_file):
    sentences = []
    last_sentence = []
    
    with open(input_file, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        word_info = line.split("\t")
        if len(word_info) >= 2:
            word_str = word_info[0].strip()
            capitalized = word_str[0].isupper()
            pos = word_info[1].strip()
            if len(word_info) >= 3:
                tag = word_info[2].strip()
            else:
                tag = None
            word = EnglishWord(word=word_str, pos=pos, capitalization_or_not=capitalized, tag=tag)
            last_sentence.append(word)
        else:
            if len(last_sentence) > 0:
                sentences.append(last_sentence)
            last_sentence = []
            sentences.append(None)
    if len(last_sentence) > 0:
        sentences.append(last_sentence)
    return sentences

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))


# ... Rest of the code remains unchanged ...

#SPECIAL_DICTIONARY = {'foreign', '$', 'or', ',', '10', 'to', 'a', 'and', 'only', 'Mich.', 'about', 'the' , 'this', 'that', 'last', 'well'}

PREFIXES_SUFFIXES = {"un", "re", "in", "dis", "en", "non", "ed", "ing", "ly", "er", "tion", "s", "es"}

#SPECIAL_DICTIONARY = {'foreign', '$', 'or', ',', '10', 'to', 'a', 'and', 'only', 'Mich.', 'about', 'the' , 'this', 'that', 'last', 'well'}.union(STOPWORDS).union(PREFIXES_SUFFIXES)

SPECIAL_DICTIONARY = set(nltk.corpus.stopwords.words('english'))

class WordFeatures:
    def __init__(self, 
                 word,
                 stem,
                 pos,
                 position,
                 sentence_length,
                 avg_word_length,
                 is_in_special_dict,
                 previous_2_pos=None,
                 previous_2_word=None,
                 previous_2_stem=None,
                 previous_pos=None,
                 previous_word=None,
                 previous_stem=None,
                 next_pos=None,
                 next_word=None,
                 next_stem=None,
                 next_2_pos=None,
                 next_2_word=None,
                 next_2_stem=None,
                 capitalized=False,
                 tag=None,
                 suffix=None,
                 prefix=None,
                 is_numeric=False,
                 has_hyphen=False,
                 has_apostrophe=False,
                 is_uppercase=False,
                 word_shape=None,  # Add this
                 bigram=None,      # Add this
                 trigram=None,     # Add this
                 has_hyphen_attr=False,
                 char_bigrams=None,
                 is_date_pattern=None,
                 is_phone_pattern=None,
                 is_common_word=None,
                 char_trigrams=None
                 ):
        self.is_common_word = is_common_word
        self.is_phone_pattern = is_phone_pattern
        self.is_date_pattern = is_date_pattern
        self.char_trigrams = char_trigrams
        self.char_bigrams = char_bigrams
        self.word_shape = word_shape      # Add this
        self.bigram = bigram              # Add this
        self.trigram = trigram 
        self.word = word
        self.stem = stem
        self.pos = pos
        self.position = position
        self.sentence_length = sentence_length
        self.avg_word_length = avg_word_length
        self.is_in_special_dict = is_in_special_dict
        self.previous_2_pos = previous_2_pos
        self.previous_2_word = previous_2_word
        self.previous_2_stem = previous_2_stem
        self.previous_pos = previous_pos
        self.previous_word = previous_word
        self.previous_stem = previous_stem
        self.next_pos = next_pos
        self.next_word = next_word
        self.next_stem = next_stem
        self.next_2_pos = next_2_pos
        self.next_2_word = next_2_word
        self.next_2_stem = next_2_stem
        self.capitalized = capitalized
        self.tag = tag
        self.suffix = suffix
        self.prefix = prefix
        self.is_numeric = is_numeric
        self.has_hyphen = has_hyphen
        self.has_apostrophe = has_apostrophe
        self.is_uppercase = is_uppercase
        self.has_hyphen_attr = has_hyphen_attr
    def __str__(self):
        return f"WordFeatures({self.word}, {self.stem}, {self.pos}, ...)"  # Truncated for brevity

import re

def word_shape(word):
    """Generate a simplified version of the word to represent its shape."""
    word = re.sub('[A-Z]', 'U', word)
    word = re.sub('[a-z]', 'L', word)
    word = re.sub('[0-9]', 'D', word)
    return word

def get_word_features(sentence):
    word_features = []
    sentence_len = len(sentence)
    avg_word_length = sum(len(word.word) for word in sentence) / sentence_len
    
    for i in range(sentence_len):
        word = sentence[i]
        position = i / sentence_len
        word_str = word.word.lower()
        word_stem = stemmer.stem(word_str)
        word_pos = word.pos
        word_capitalized = word.capitalization_or_not
        word_tag = word.tag
        is_in_special_dict = word_str in SPECIAL_DICTIONARY

        # Enhanced word shape
        word_shape_str = word_shape(word_str)

        # Morphological features
        suffix = word_str[-3:]  # last three characters
        prefix = word_str[:3]  # first three characters

        # Boolean features
        is_numeric = word_str.isdigit()
        has_hyphen = '-' in word_str
        has_apostrophe = '\'' in word_str
        is_uppercase = word_str.isupper()

        # Character N-grams
        char_bigrams = [word_str[i:i+2] for i in range(len(word_str) - 1)]
        char_trigrams = [word_str[i:i+3] for i in range(len(word_str) - 2)]

        # Regular expressions to capture patterns
        is_date_pattern = bool(re.match(r"\d{2}/\d{2}/\d{4}", word_str))
        is_phone_pattern = bool(re.match(r"\d{3}-\d{3}-\d{4}", word_str))

        # Check if the word is a common English word
        is_common_word = word_str in nltk.corpus.words.words()

        # Previous-2 word features
        previous_2_word_str = previous_2_word_pos = previous_2_word_stem = None
        if i >= 2:
            previous_2_word = sentence[i-2]
            previous_2_word_str = previous_2_word.word
            previous_2_word_pos = previous_2_word.pos
            previous_2_word_stem = stemmer.stem(previous_2_word_str)

        # Previous word features
        previous_word_str = previous_word_pos = previous_word_stem = None
        if i >= 1:
            previous_word = sentence[i-1]
            previous_word_str = previous_word.word
            previous_word_pos = previous_word.pos
            previous_word_stem = stemmer.stem(previous_word_str)

        # Next word features
        next_word_str = next_word_pos = next_word_stem = None
        if i <= sentence_len - 2:
            next_word = sentence[i+1]
            next_word_str = next_word.word
            next_word_pos = next_word.pos
            next_word_stem = stemmer.stem(next_word_str)

        # Next-2 word features
        next_2_word_str = next_2_word_pos = next_2_word_stem = None
        if i <= sentence_len - 3:
            next_2_word = sentence[i+2]
            next_2_word_str = next_2_word.word
            next_2_word_pos = next_2_word.pos
            next_2_word_stem = stemmer.stem(next_2_word_str)

        features = WordFeatures(
            word=word_str, 
            stem=word_stem, 
            pos=word_pos,
            position=position,
            sentence_length=sentence_len,
            avg_word_length=avg_word_length,
            is_in_special_dict=is_in_special_dict,
            word_shape=word_shape_str,
            suffix=suffix,
            prefix=prefix,
            is_numeric=is_numeric,
            has_hyphen_attr=has_hyphen,
            has_apostrophe=has_apostrophe,
            is_uppercase=is_uppercase,
            char_bigrams=char_bigrams,
            char_trigrams=char_trigrams,
            is_date_pattern=is_date_pattern,
            is_phone_pattern=is_phone_pattern,
            is_common_word=is_common_word,
            previous_2_pos=previous_2_word_pos,
            previous_2_word=previous_2_word_str,
            previous_2_stem=previous_2_word_stem,
            previous_pos=previous_word_pos,
            previous_word=previous_word_str,
            previous_stem=previous_word_stem,
            next_pos=next_word_pos,
            next_word=next_word_str,
            next_stem=next_word_stem,
            next_2_pos=next_2_word_pos,
            next_2_word=next_2_word_str,
            next_2_stem=next_2_word_stem,
            capitalized=word.capitalization_or_not,
            tag=word.tag
        )
        word_features.append(features)
    
    return word_features



    

def main():
    parser = argparse.ArgumentParser(
        description="A feature selector for Maxent Noun Group tagger.")
    parser.add_argument("inputfile", help="input corpus file")
    parser.add_argument("outfile", help="feature selection output file")

    args = parser.parse_args()

    sentences = parse(args.inputfile)

    print('start writing')
    with open(args.outfile, "w") as f:
        for sentence in sentences:
            if sentence is None:
                f.write("\n")
            else:
                word_features = get_word_features(sentence)
                for word_feature in word_features:
                    word_feature_str_list = []
                    for key, value in vars(word_feature).items():
                        if value is None or key == "word" or key == "tag":
                            continue
                        word_feature_str_list.append(f"{key.upper()}={value}")
                    word_feature_str_list.insert(0, word_feature.word)
                    if word_feature.tag is not None:
                        word_feature_str_list.append(word_feature.tag)
                    f.write("\t".join(word_feature_str_list))
                    f.write("\n")
        print(f"{args.inputfile} -> {args.outfile}.")

if __name__ == '__main__':
    main()



