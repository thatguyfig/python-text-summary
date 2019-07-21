import argparse
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from nltk.probability import FreqDist
from heapq import nlargest
from collections import defaultdict

def main():

    # first of all, lets parse the arguments from command-line
    args = parse_arguments()

    # next, read in the file content
    content = read_file(args.filepath)

    # remove stop words and punctuation
    content = sanitize_input(content)

    # tokenize sentences and words
    sentence_tokens, word_tokens = tokenize_content(content)

    # rank sentences based on their content
    sentence_ranks = score_tokens(word_tokens, sentence_tokens)

    # summarize the text and print it
    print(summarize(sentence_ranks, sentence_tokens, args.length))

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath',help='Filename of the text to summarize')
    parser.add_argument('-l','--length', default=4, help='Number of sentences to return in the summary output')
    args = parser.parse_args()
    return args

def read_file(path):
    try:
        with open(path, 'r') as file:
            return file.read()
    except IOError as e:
        print("Fatal Error: File ({}) could not be located or is not readable\n".format(path), e)

def sanitize_input(data):
    replace = {
        ord('\f') : ' ',
        ord('\t') : ' ',
        ord('\n') : ' ',
        ord('\r') : None
    }
    return data.translate(replace)

def tokenize_content(content):
    stop_words = set(stopwords.words('english') + list(punctuation))
    words = word_tokenize(content.lower())

    return [
        sent_tokenize(content),
        [word for word in words if word not in stop_words]
    ]

def score_tokens(filtered_words, sentence_tokens):
    word_freq = FreqDist(filtered_words)

    ranking = defaultdict(int)

    for i, sentence in enumerate(sentence_tokens):
        for word in word_tokenize(sentence.lower()):
            if word in word_freq:
                ranking[i] += word_freq[word]
    
    return ranking

def summarize(ranks, sentences, length=1):
    if int(length) > len(sentences):
        print("Error, more sentences requested than available. Use --l (--length) flag to adjust.")
        exit()
    
    indexes = nlargest(int(length), ranks, key=ranks.get)
    final_sentences = [sentences[j] for j in sorted(indexes)]
    return ' '.join(final_sentences)


if __name__ == "__main__":
    main()