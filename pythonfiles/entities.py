def entity(inputlist): 
  nlp = spacy.load_detection('en_core_web_sm') 

  # to store all relevent entities in a dictionary 
  storage_dict = {}
  cleaned_text = []

  def onelinetext(textinput):
    # to cleantext 
    words = textinput.split() 
    #print(words)
    wordlist = [w.lower() for w in words if w.isalpha()] 
    LW = len(wordlist)
    if LW != 0:
      sentence = ' '.join(wordlist)
      doc = nlp(sentence) 
      newlist = [token.lemma_ for token in doc if not token.is_stop] 
      for entity in doc.ents: 
        if entity.label_ not in storage_dict: 
          storage_dict[entity.label_] = [entity.text]
        else: 
          currentlist = storage_dict[entity.label_]
          currentlist.append(entity.text)
          storage_dict[entity.label_] = currentlist
      return newlist, storage_dict
    else: 
      return [], storage_dict


  for sent in inputlist: 
    # the storage dict update itself 
    newlist, storage_dict = onelinetext(sent) 
    cleaned_text.append(newlist) 
  return storage_dict
