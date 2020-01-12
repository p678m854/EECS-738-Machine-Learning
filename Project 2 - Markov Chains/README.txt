Poject 2, Markov Chains
Author: Patrick McNamee
Partner: Tim Fox

Objective: Form a Hidden Markov Method of text prediciton of Shakespeare plays

Hidden States: Actors who speak the lines
Transition: Which actor speaks after what line (generally the same character as they speak multiple lines)
Emissions: What words do the actors speak.

How did we do this: First we trained on either the entire works of Shakespeare or one of his plays, King Henry IV, simple to save time on training and testing. We went through all the lines and created a Markov model of what actor spoke by recording the frequency. We also created a first and second order Markov model for text prediction of the lines using dictionaries specific to the actors to record the frequency of each word. After we went through the entirety of the corpus, we normalized the distribution using Maximum Likelihood methods.
Training Files: Shakespeare_data.csv (we support it but the Viterbi Algorithm really slows down) and "Henry IV.csv"

Testing: We gave a Viterbi algorithm two text files with only lines and no information about what actor was speaking. The algorithm return the most likely state path (i.e. what actors delivered the lines) and then using the frequencies stored in the dictionary delivered the next line using Maximum Likelihood methods, i.e. what had the highest likelihood.
Test files: firstLinesHenryIV.txt (Line 297) and henryIV13.txt (Line 301)

How to run:
  1) In train.py, adjust training file adresses at line 7 and Viterbi Algorithms at lines 297 and 301 to whatever you local storing of them is. I defaulted so to the local directory
  2) Run train.py
  3) If you want to kill some computer time, change training file to Shakespeare_data.csv and then rerun train.py.


Comments:
  1) Maximum Likelihood -- you have to test our Markov Models with only words or combinations of words that exist in Shakespeare. Otherwise there will be a zero propability for all actors that they spoke it. You could get around this using a smoothing method when dealing with normalizing the frequency.
  2) Between line dependency -- We did not deal with lines being interconnected and instead as independent. You could potentially have a Markov Model, order > 1, that could do that prediction across lines but this interfered with our assumption of hidden states being actors. It was also difficult to implement with all the scene transitions and play transitions. Hence we can only project forward one line in text prediciton using our hidden states. 
