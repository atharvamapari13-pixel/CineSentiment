# Step 1 : Import Required Libraries

from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences


###############################################################
# Step 2: Configuration of Values
###############################################################

VOCAB_SIZE = 10000 # Consider most Frequent 10000 unique words
MAX_LENTH = 200 # Consider Maximum 200 words in reviews

###############################################################
# Step 3: Load the IMDB Databases (Internet Movie Databases)
###############################################################
print("-"*40)
print("Movie Review Sentiment Analysis Using LSTM")
print("-"*40)

print("Loading The Dataset....")

(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words= VOCAB_SIZE)

print("IMDB dataset loaded successfully")

print("Number of Training reviews :",len(X_train))
print("Number of Test reviews :",len(X_test))


##########################################################################
# X_train    Reviews used for training 
# Y_train    Actual Sentiments of training
# X_test     Review used for tesing
# Y_test     Actual Sentiment of testing
##########################################################################

# Positive = 1
# Negative = 0


###############################################################
# Step 4: Load the word  Dictotionary (Internet Movie Databases)
###############################################################

word_index = imdb.get_word_index()


# Dictionary contains Mapping of word and its corresponding number
# Drishyam is Good movie      -> (20,56,78,43)
# 20 -> Drishyam
# 56 -> is
# 78 -> Good
# 43 -> Movie


###############################################################
# Step 5: Create a reverse Dictionary
###############################################################

reverse_word_index = {}

for word, index in word_index.items():
    reverse_word_index[index+3] = word

###############################################################
# Step 6: Function to Decode the review (number to word)
###############################################################

def DecodeReview(encoded_review):
    words = []
    for number in encoded_review:
        if number>=3:   # Ignore first 3
            word = reverse_word_index.get(number,"?")
            words.append(word)
    return " ".join(words) # Join the list of words

###############################################################
# Step 7: Display Sample Reviews
###############################################################


print("-"*40)
print("--------------Sample Reviews-------------")
print("-"*40)

for i in range(3,7):
    review = DecodeReview(X_train[i])
    print("-"*40)

    print("Review Number :",i+1)
    print("Review:")
    print(review)
    print("-"*40)


    if Y_train[i] == 1:
        print("Sentiment : positive")
    else:
        print("Sentiment : Negative")


###############################################################
# Step 8: Padding
###############################################################


X_train_padded = pad_sequences(
    X_train,
    maxlen = MAX_LENTH
)

X_test_padded = pad_sequences(
    X_test,
    maxlen = MAX_LENTH
)

print("Trining Data Shape :",X_train_padded.shape)
print("Testing Data Shape :",X_test_padded.shape)

###############################################################
# Step 9: Create LSTM Model
###############################################################

model = Sequential()
model.add(
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim= 32  # Each word is Represented in 32 values
    )
)

model.add(
    LSTM(
        units = 64 # Size of LSTM Hidden State
    )
)

model.add(
    Dense
    (
        units = 1,   # One Output 
        activation="sigmoid"  # Used To produce Probilitys
    )
)

# Project Architecture
# Review -> Embedding -> LSTM -> Dense -> Sigmoind -> Positive / Negative

###############################################################
# Step 10: Compile The model
###############################################################


model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("Model Compiled Successfully")


###############################################################
# Step 11: Train The Model
###############################################################

print("Model Training")

history = model.fit(
    X_train_padded,
    Y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.2
)

print("Model Training Completed")


###############################################################
# Step 12: Evaluate The Model
###############################################################

test_loss, test_accuracy = model.evaluate(
    X_test_padded,
    Y_test,
    verbose=0
)

print("Testing Loss :", test_loss)
print("Testing Accuracy :", test_accuracy)
###############################################################
# Step 13: Predict the Review
###############################################################

TEST_REVIEW_NUMBER = 0
original_review = X_test[TEST_REVIEW_NUMBER]
decoded_review = DecodeReview(original_review)

print("Review Given to the model:")
print(decoded_review)


###############################################################
# Step 14: Get the Actual Sentiment
###############################################################

actual_value = Y_test[TEST_REVIEW_NUMBER]

if actual_value == 1:
    actual_sentiment = "POSITIVE"
else:
    actual_sentiment = "NEGATIVE"
print("Actual Sentiment :",actual_sentiment)

###############################################################
# Step 15: Predict the sentiment
###############################################################

review_for_prediction = X_test_padded[TEST_REVIEW_NUMBER : TEST_REVIEW_NUMBER + 1]
prediction = model.predict(
    review_for_prediction,
    verbose = 0
)


probabality = prediction[0][0]
if probabality >= 0.5 :
    predicted_sentiment = "POSITIVE"
else:
    predicted_sentiment = "NEGATIVE"

print("Final")
print("-"*40)
print("Prediction Probablity:", probabality)
print("Actual Sentiment:",actual_sentiment)
print("Predicted Sentiment:",predicted_sentiment)
print("-"*40)



model.save("Final Model.keras")



