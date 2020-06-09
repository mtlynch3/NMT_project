from seq2seq import *
import torch

from nltk.translate.bleu_score import sentence_bleu


def evaluateFromInput(encoder, decoder):
    #get input setence, pair[0] in example above
    print()

    print("Enter a sentence: ")
    input_sentence = input(">> ")
    print()

    #normalize input sentence
    input_sentence = normalizeString(input_sentence)
    
    output_words, attentions = evaluate(encoder, decoder, input_sentence)
    output_sentence = ' '.join(output_words)

    print("Translation: ")
    print(">> ", output_sentence)

    print()

#assumes eng to spa when reverse=False
def get_new_pairs(reverse=False):
    print("Reading lines for new pairs...")

    # Read the file and split into lines
    sp_lines = open('newSpanish.txt', encoding='utf-8').\
        read().strip().split('\n')
    en_lines = open('newEnglish.txt', encoding='utf-8').\
        read().strip().split('\n')
    # Split every line into pairs and normalize
    new_pairs = []
    for i in range(len(sp_lines)):
        new_pairs.append([normalizeString(en_lines[i]), normalizeString(sp_lines[i])])

    # Reverse pairs
    if reverse:
        new_pairs = [list(reversed(p)) for p in new_pairs]

    return new_pairs

def evalRandomNew(encoder, decoder, n=10):
    new_pairs = get_new_pairs()
    for i in range(n):
        pair = random.choice(new_pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

        target = [pair[1].split()]
        evalBLEU(output_words, target)

def evalBLEU(candidate, reference):
    print('Cumulative 1-gram: %f' % sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
    print('Cumulative 2-gram: %f' % sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
    

encoder = torch.load("encoder-eng-spa-OpenSub.pt",map_location='cpu')
decoder = torch.load("decoder-eng-spa-OpenSub.pt",map_location='cpu')

evaluateRandomly(encoder, decoder, 10)
