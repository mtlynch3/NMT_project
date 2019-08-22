from seq2seq import *
import torch

print("HELLO BITCH")


encoder = torch.load("encoder-eng-spa-OpenSub.pt",map_location='cpu')
decoder = torch.load("decoder-eng-spa-OpenSub.pt",map_location='cpu')

evaluateRandomly(encoder, decoder, 20)

"""
continueProgram = True

while continueProgram:
    print("E2S or S2E?")
    command = input(">> ")
    if command == "E2S":
        print("ok. enter an english sentence below:")
        input_sentence = input(">> ")

        output_words, attentions = evaluate(E2S_encoder, E2S_decoder, input_sentence)
        output_sentence = ' '.join(output_words)

        print("translation:")
        print(">> ", output_sentence)
    elif command == "exit":
        continueProgram = False
    else:
        print("stupid hoe!!!!!!")
        
    
    
    
"""
