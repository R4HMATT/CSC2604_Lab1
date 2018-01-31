
I believe a large problem with finding the cosine similarity between the word pairs with judged similarity and the
our bigram generated word vectors is the behaviour of word pairs on high ends of the judged similarity
in relation to a contextual model.

If two words are very high on the judged similarity scale, they are essentially synonyms that often replace each other in sentences, as using one directly after the other would be redundant.
This means their bigram readings will be very low. The definition for context used in Rubenstein and Goodenough's paper was all the words that
occurred in a word's sentence set, which circumvents the problem of the most similar words being unlikely to appear in a bigram.

Therefore, I propose a model that utilizes word pairs that have a judged similarity with values somewhat higher than intermediate (2.0-3.5). The justification for this is that these word pairs are similar enough to perhaps relate to the same topic, but would not be effective synonyms to replace each other. As a result we have words that are more likely to appear together as each others contexts, not simply in each others contexts. In addition to this, using larger n-grams in the context may
improve accuracy, as synonyms may not appear next to each other often, but may be more likely to appear in the same sentence. This is what was noticed in Rubenstein and Goodenough for sentence sets, so there is an indication that increasing the size of the context window may help.

Unfortunately we did not have time to implement these ideas to see if they did improve correlation, but we would like to hear if there's anything wrong with our reasoning.

