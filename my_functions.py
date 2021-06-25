def create_corpus_file(csv_filepath, output_filepath):
    """Dataset Processing function, part of Dimitris Kaplanis' thesis project.
    
    This function takes 2 arguments:
    
    csv_filepath: string - the csv path 
    output_filepath: string - your chosen output file name, please include .txt
    
    This function creates a corpus text file by processing a csv file and outputting the column (of the body)
    to a text file and adds a line delimited (\n) to separate each piece of text.
    
    This was required for the huge reddit database file that could not otherwise be properly processed and 
    passed onto word2vec for training.
    """    
    import csv
    import time
    start_time = time.time()
    with open(csv_filepath, encoding = 'utf-8') as csvfile:
        datareader = csv.reader(csvfile)
        count = 0
        header = next(csvfile)
        print(time.asctime(time.localtime()), " ---- Beginning Processing")
        with open(output_filepath, 'w+') as output:
            # Check file as empty
            if header != None:
                for row in datareader:
                        # Iterate over each row after the header in the csv
                        # row variable is a list that represents a row in csv
                    processed_row = str(' '.join(row)) + '\n'
                    output.write(processed_row)
                    count += 1
                    if count == 1000000:
                        print(time.asctime(time.localtime()), " ---- Processed 1,000,000 Rows of data.")
                        count = 0
    print('Processing took:', int((time.time()-start_time)/60), ' minutes')
    output.close()
    csvfile.close()
	
def process_dataset(data, body_col, cleaned_col):
    """Dataset Processing function, part of Dimitris Kaplanis' thesis project.
    
    This function takes 3 arguments:
    
    data: the pandas dataframe path that contains our dataframe
    body_col: string - the name of the column containing the body of transcripts
    cleaned_col: string - the name you want to give the column created containing the cleaned text
    
    This function also reduces our labels from 1-2-3 to 0 and 1 for Non-conspiratorial and Conspiratorial.
    It also creates two new columns "Consp" and "NonConsp" because our model will produce a two vector output for our
    binary classification problem.   
    
    It also tokenizes the cleaned column for further processing down the line.
    
    Example use:
    >>>>>>>>>>>> df = process_dataset(data, '2', '2-cleaned')
    """
    import string
    import time
    import random
    import os
    import re
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    stoplist = stopwords.words('english')
    start_time = time.time()
    data_dir = os.getcwd()
    pat = r'\b(?:{})\b'.format('|'.join(stoplist)) # regex pattern for replacement
    def remove_punct(text):
        text_nopunct = ''
        text_nopunct = re.sub('['+string.punctuation+']', '', str(text.lower())) # Change .lower() if it doesn't work here
        return text_nopunct
    data[cleaned_col] = data[body_col].apply(lambda x: remove_punct(str(x)))
    data[cleaned_col] = data[cleaned_col].str.replace(pat, '')
    data[cleaned_col] = data[cleaned_col].str.replace(r'\s+', ' ')
    cols = ['1', body_col, cleaned_col, '3']
    data = data[cols]
    data = data[data['3'] != 'x']
    data = data.replace({'3': {'1': 1, '2': 2, '3': 3}})
    data = data.replace({'3': {1: 0, 2: 1, 3: 1}}) # Convert 1 to 0 (Non-consp) and 2-3 to 1 (Conspiratorial) for binary class.
    consp = []
    non_consp = []
    for label in data['3']:
        if label == 0:
            consp.append(0)
            non_consp.append(1)
        elif label == 1:
            consp.append(1)
            non_consp.append(0)
    data['Consp'] = consp
    data['NonConsp'] = non_consp
    tokens = [word_tokenize(sen) for sen in data[cleaned_col]]
    data['tokens'] = tokens
    data = data[['1', body_col, cleaned_col, 'tokens', '3', 'Consp', 'NonConsp']]
    data.to_csv(str(data_dir) + '/combined_data_cleaned' + str(random.randrange(0, 100)) + '.csv', mode='a', index=None, header=True)
    end_time = time.time()
    print("Processing completed in:", int(end_time-start_time), "seconds.")
    return data
	
def reddit_data_cleaning(filepath, batchsize=20000):
    """Dataset Processing function, part of Dimitris Kaplanis' thesis project.
    
    This function takes 3 arguments:
    
    filepath: the pandas dataframe path that contains our dataframe
    batchsize: integer - the amount of rows you want to process per batch. Faster computers can handle more rows.
                         For slower machines, opt for ~20,000-30,000. Machines with more RAM can cope with batches
                         of over 50-150k rows.
        
    This function takes care of all the pre-processing we have opted to do for the Reddit Data that is then used to create
    our conspiracy corpus.
    Pre-processing includes removing punctuation, lowering the text (de-capitalizing), removing stopwords using NLTK's
    English stopwords list.
    Regex is used to clear the text of some other special characters.
    Tokenization happens using NLTK's tex
    
    It also tokenizes the cleaned column for further processing down the line.
    """
    import pandas as pd
    import re
    import time
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    stoplist = stopwords.words('english')
    df_for_size = pd.read_csv(filepath, encoding='utf-8')
    total_batches = int(df_for_size.shape[0])/int(batchsize)
    if batchsize:
        df = pd.read_csv(filepath, encoding='utf-8', error_bad_lines=False, chunksize=batchsize, iterator=True, lineterminator='\n')
    print("Beginning the data cleaning process!")
    start_time = time.time()
    flag = 1
    def remove_punct(text):
        text_nopunct = ''
        text_nopunct = re.sub('['+string.punctuation+']', '', str(text.lower())) # Change .lower() if it doesn't work here
        return text_nopunct
    chunk_num = 1
    total_batches = int(df_for_size.shape[0])/int(batchsize)
    for chunk in df:
        print("Beginning processing of a chunk #: ", str(chunk_num), " of ", str(total_batches))
        print("Removing punctuation and stopwords...")
        pat = r'\b(?:{})\b'.format('|'.join(stoplist)) # regex pattern for replacement
        chunk[u'body'] = chunk[u'body'].apply(lambda x: remove_punct(str(x)))
        chunk[u'body'] = chunk[u'body'].str.replace(pat, '')
        chunk[u'body'] = chunk[u'body'].str.replace(r'\s+', ' ')
        chunk[u'body'].apply(lambda x: [item for item in x if item not in stoplist])
        print("Tokenizing.")
        chunk[u'tokens'] = chunk[u'body'].apply(word_tokenize)
        chunk[u'tokens'] = chunk[u'tokens'].str.lower()
        chunk_num += 1
        if flag == 1:
            print("Beginning writing a new file...")
            chunk.dropna(how='any')
            chunk = chunk[chunk['body_cleaned'] != 'deleted']
            chunk = chunk[chunk['body_cleaned'] != 'removed']
            chunk.to_csv(str(filepath + '_processed.csv'), mode='a', index=None, header=True) # New file requires header
            flag = 0
        else:
            print("Adding a chunk into an already existing file...")
            chunk.dropna(how='any')
            chunk = chunk[chunk['body_cleaned'] != 'deleted']
            chunk = chunk[chunk['body_cleaned'] != 'removed']
            chunk.to_csv(str(filepath + '_processed.csv'), mode='a', index=None, header=None) # Existing file - does not require header
    end_time = time.time()
    print("Processing has been completed in: ", ((end_time - start_time)/60), " minutes.")
    print("Processed ", chunk_num, " chunks in total.")

def train_word2vec(corpus_file, output_model_name, dimentionality=300, min_word_count=5, context=5, compute_loss=True, sg_or_cbow=1):
    """Word2Vec training function, part of Dimitris Kaplanis' thesis project.
    
    This function takes 7 arguments:
    
    Dimensionality: int - depth of our model. 200 or 300 are the configurations used
    min_word_count: int - the amount of times a word has to show up in the corpus to be kept. 
    context: int - how many words to the left and right should our model consider when generating vectors
    compute_loss: bool - computes loss for the purposes of logging our model's performance.
    corpus_file: str - either a string path to the file that is made by create_corpus_file(), or
                 a link to the tokens column of a pandas dataframe, e.g. df['tokens'].
    sg_or_bow: What algorithm should Word2Vec use to generate word embeddings? 1 for Skip-Gram, 0 for Continuous Bag of Words.
    
    Once this function is ran, the model will train and output all logging verbose.
    Once complete, it will save a file at the same directory.
    """
    import gensim
    import pandas as pd
    from gensim.models import word2vec
    import multiprocessing
    start = time.time()
    
    # Get updates on progress using logging
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
        level=logging.INFO)
    # Hyperparameter tuning
    dimensionality = dimentionality    # Dimensionality of the hidden layer representation
    min_word_count = min_word_count   # Minimum word count to keep a word in the vocabulary
    num_workers = multiprocessing.cpu_count() 
    context = context          # Context window size (left and right of each word)                                                   
    downsampling = 1e-3   # Downsample setting for frequent words
    sg = 1 # 1 for skip-gram, 0 for CBOW
    compute_loss = compute_loss # computes and saves the last loss function

    # Initialize and train the model. 
    print("Training model...");
    model = word2vec.Word2Vec((corpus_file), workers=num_workers, vector_size=dimentionality, min_count = min_word_count, window = context, sample = downsampling, compute_loss = compute_loss, sg = sg)

    # We stop training after it is done, and call init_sims to make the model more memory efficient
    # by normalizing the vectors (in-place).
    model.init_sims(replace=True);

    # Save the model
    model_name = output_model_name
    model.save(model_name)

    print('Total time: ' + str(((time.time() - start)/60)) + ' minutes')
	
def load_thesis_datasets():
    """Dataset Processing function, part of Dimitris Kaplanis' thesis project.
    
    This function takes no arguments.
    It simply imports the 4 datasets gathered by master's students of this and past groups from the current
    working directory.
    """
    import pandas as pd
    import os
    path = os.getcwd()
    # Retrieve datasets based on their pre-assigned names
    dataset1 = path + '/Alfano_Cleaned.csv'
    dataset2 = path + '/First_Cohort_Cleaned.csv'
    dataset3 = path + '/Second_Cohort_Cleaned.csv'
    dataset4 = path + '/Spring 2021_Cleaned.csv'

    # Rename columns 
    column_names = {

        'Unnamed: 0' : '0',
        '0':'1',
        '1':'2',
        '2':'3'
    }

    dataset1 = pd.read_csv(dataset1).rename(columns = column_names, inplace = False)
    dataset2 = pd.read_csv(dataset2).rename(columns = column_names, inplace = False)
    dataset3 = pd.read_csv(dataset3).rename(columns = column_names, inplace = False)
    dataset4 = pd.read_csv(dataset4).rename(columns = column_names, inplace = False)

    frames = [dataset1, dataset2, dataset3, dataset4]

    dataframe = pd.concat(frames)
    return dataframe
	
def generate_word2vector(tokens, vector, generate_missing=False, dimensionality=300):
    """
    Vector Generation Function, part of Dimitris Kaplanis' thesis project.
    
    The function takes 4 arguments:
    tokens: str - path to tokens file or column in a pandas dataframe.
    vector: variable that leads to model.wv
    generate_missing: bool - whether you want to generate a random number for missing words or replace with 0.
    dimensionality: depth of our vector representation.
    
    Returns an averaged 1D vector that can be processed by our classifiers.
    """
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(dimensionality) for word in tokens]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(dimensionality) for word in tokens]
    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    onedvec = np.divide(summed, length) # We average to generate a 1D vector that our classifiers can work with.
    return onedvec
	
def generate_word_embeddings(vectors, processed_text, generate_missing=False):
    """
    Vector Generation Function, part of Dimitris Kaplanis' thesis project.
    
    The function takes 3 arguments:
    vectors: variable that leads to model.wv
    processed_text: dataframe or file that contains a row called 'tokens' (should exist if we have preprocessed our dataframe)
    generate_missing: bool - whether you want to generate a random number for missing words or replace with 0.
        
    Returns a list of word embeddings.
    """
    from my_functions import generate_word2vector
    import pandas as pd
    word_embeddings = processed_text['tokens'].apply(lambda x: generate_word2vector(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(word_embeddings)
	
def setup_rnn(word_embeddings, max_sequence_length, num_words, embedding_dim, labels_index):
    
    """
    Recurrent Neural Network setup function, part of Dimitris Kaplanis' thesis project.
    
    The function takes 5 arguments:
    word_embeddings: vector representation of our corpus
    processed_text: dataframe or file that contains a row called 'tokens' (should exist if we have preprocessed our dataframe)
    generate_missing: bool - whether you want to generate a random number for missing words or replace with 0.
        
    Returns a list of word embeddings.
    """
    from keras.callbacks import ModelCheckpoint
    from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Embedding, LeakyReLU
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential
    from keras.models import Model
    
    embedding_layer = Embedding(num_words,
                            embedding_dim,
                            weights=[word_embeddings],
                            input_length=max_sequence_length,
                            trainable=False)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    lstm = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(embedded_sequences)
    x = Dense(128, activation='relu')(lstm)
    x = Dropout(0.5)(x)
    preds = Dense(labels_index, activation='sigmoid')(x) #benchmark softmax
    model = Model(sequence_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    return model
	
def generate_rnn_model(label_names, emb_weights, max_text_length, word_index, epochs, dimensionality=300):
    """
    Recurrent Neural Network generation function, part of Dimitris Kaplanis' thesis project.
    
    The function takes 5 arguments:
    emb_weights: str or var - link to file or vector of our corpus
    max_text_length: int - maximum sentence length (for our dataframe it was ~21000)
    dimensionality: int - depth of our model
    label_names: list - list of strings of our label names (for our dataframe ['Consp', 'NonConsp'])
    epochs: int - the number of epochs the model will train for.
        
    Initializes the training of the RNN model and outputs its progress.
    """
    from my_functions import setup_rnn
    
    model = setup_rnn(emb_weights, max_text_length, len(word_index)+1, dimensionality, len(list(label_names)))
    
    print(model)
    
    run = model.fit(x_train, y_train, epochs=epochs, validation_split=0.3, shuffle=True, batch_size=batch_size)
    
	
def dataset_shuffler(dataframe, body_column, label_column, test_size):
    import sklearn
    from sklearn.model_selection import train_test_split
    Train_X, Test_X, Train_Y, Test_Y = train_test_split(dataframe[body_column], dataframe[label_column],test_size=test_size, random_state=42)
    return Train_X, Test_X, Train_Y, Test_Y