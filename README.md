To run evaluation of existing models:
  
  **python3 -W ignore loaded.py language-option**
  
  
Warnings are ignored for BLEU score calculations

loaded.py should be modified for whatever evaluation functions you want to use

If you just want to play around I recommend using evaluateRandomly or evaluateFromInput :)

 
To train new models:

  **python3 seq2seq.py language-option**
  

language-option: e2s is English to Spanish, s2e is Spanish to English

Must use pytorch v 1.0.0
