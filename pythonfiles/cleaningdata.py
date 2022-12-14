class cleandata:
  def __init__(self, inputdataframe): 
    self.dataframe = inputdataframe 
    self.bookreview = inputdataframe['book review'] 
    self.rating = inputdataframe['rating'] 
    self.wordcount = inputdataframe['number of words'] 

  def Cleaningdata(self, rating): 
    nlp = spacy.load('en_core_web_sm')  

    def cleanData(doc, stemming = False):
      # to include a build function to do text data cleaning 
      # to check if the words are stop words 
      tokens = [tokens for tokens in doc if (tokens.is_stop == False)]
      tokens = [tokens for tokens in tokens if (tokens.is_punct == False)]
      # to find the stem of the words 
      final_token = [token.lemma_ for token in tokens]
      # to put the words together into a pharaphrase again 
      return " ".join(final_token)

    # to count the number of element 
    L = len(self.rating)
    # to store the texts in a list
    textcollectionlist = []

    for i in range(0, L): 
      currentrating = self.rating[i] 
      if currentrating == rating:
        textcollectionlist = textcollectionlist + self.bookreview[i].split()
    # to clearn the entire data 
    # the intention is to use the entire corpus for text mining later 
    newtest = ' '.join(textcollectionlist) 
    self.cleaned_text = cleanData(newtest)
