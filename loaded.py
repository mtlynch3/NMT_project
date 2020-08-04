from seq2seq import *
import torch

from nltk.translate.bleu_score import sentence_bleu


def evaluateFromInput(encoder, decoder):
    while True:
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

    new_pairs = filterPairs(new_pairs)
    # Reverse pairs
    if reverse:
        new_pairs = [list(reversed(p)) for p in new_pairs]

    return new_pairs

def prepBLEU(sentence):
    bad_words = ['','<EOS>','.',',','?','!']
    new_list = []
    for word in sentence:
        if word not in bad_words:
            new_list.append(word)
    return new_list


def evalBLEU(candidate, reference): #evaluates BLEU score for a single example
    reference = [reference]
    scores = []

    #1-gram
    scores.append(sentence_bleu(reference, candidate, weights=(1, 0, 0, 0)))
    #2-gram
    scores.append(sentence_bleu(reference, candidate, weights=(0.5, 0.5, 0, 0)))
    #3-gram
    scores.append(sentence_bleu(reference, candidate, weights=(0.33, 0.33, 0.33, 0)))
    #4-gram
    scores.append(sentence_bleu(reference, candidate, weights=(0.25, 0.25, 0.25, 0.25)))

    return scores



def testTrainingData(encoder, decoder, n=100, remove_zeroes=False): #n = number of examples; remove_zeroes=False -> true avg by default 
    scores = [0.0, 0.0, 0.0, 0.0]
    counts = [0, 0, 0, 0] #for remove_zeroes true
    #process each example
    for i in range(n):
        pair = random.choice(pairs)
        if(i%500 == 0):
            display_num = i+500
            print("Read %d examples" % display_num)
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)

        target = pair[1].split()
        output_words = prepBLEU(output_words)
        target = prepBLEU(target)
        curr_scores = evalBLEU(output_words, target)

        if remove_zeroes:
            for i in range(4):
                if(curr_scores[i] > 0.05): #good enough score
                    scores[i]+= curr_scores[i]
                    counts[i] += 1 #update numbers
        else:
            for i in range(4):
                scores[i] += curr_scores[i]
    #end examples

    for i in range(4):
        if remove_zeroes:
            scores[i] = scores[i] / counts[i]
        else:
            scores[i] = scores[i]/n 


    print(counts)
    return scores


def testNewData(encoder, decoder, n=100, remove_zeroes=False, reverse=False):
    if reverse:
        new_pairs = get_new_pairs(True)
    else:
        new_pairs = get_new_pairs()
    scores = [0.0, 0.0, 0.0, 0.0]
    counts = [0, 0, 0, 0] #for remove_zeroes true
    actual_count = 0 #account for unknown input
    #process each example
    for i in range(n):
        if(i%500 == 0):
            display_num = i + 500
            print("Read %d examples" % display_num)
            print("Processed %d examples" % actual_count)

        pair = random.choice(new_pairs)
        output_words, attentions = evaluate(encoder, decoder, pair[0])

        if output_words == None: #UNK input
            continue

        output_sentence = ' '.join(output_words)

        target = pair[1].split()
        output_words = prepBLEU(output_words)
        target = prepBLEU(target)
        curr_scores = evalBLEU(output_words, target)
        actual_count+=1

        if remove_zeroes:
            for i in range(4):
                if(curr_scores[i] > 0.05): #good enough score
                    scores[i]+= curr_scores[i]
                    counts[i] += 1 #update numbers
        else:
            for i in range(4):
                scores[i] += curr_scores[i]
    #end examples

    for i in range(4):
        if remove_zeroes:
            scores[i] = scores[i] / counts[i]
        else:
            scores[i] = scores[i]/actual_count 


    print(actual_count, counts)
    return scores
    

###########
##TESTING##
########### 

if e2s:
    #english to spanish
    encoder = torch.load("encoder-eng-spa-OpenSub.pt",map_location='cpu')
    decoder = torch.load("decoder-eng-spa-OpenSub.pt",map_location='cpu')
elif s2e:
    #spanish to english
    encoder = torch.load("encoder-spa-eng-OpenSub.pt",map_location='cpu')
    decoder = torch.load("decoder-spa-eng-OpenSub.pt",map_location='cpu')
else:
    print("could not load models :/")
    exit(1)

#evaluateRandomly(encoder, decoder, 100)
evaluateFromInput(encoder, decoder)

#print("Averages: ", testTrainingData(encoder, decoder, 25000, True))
#print("Averages: ", testNewData(encoder, decoder, 25000, True, s2e)) #last parameter for if spanish to english



 