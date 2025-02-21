import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from pandarallel import pandarallel
import string
import numpy as np


dataset = pandas.read_csv("parsed_files/Home_and_Kitchen.csv", lineterminator='\n') 

# use existing csv
dataset = dataset[dataset['reviewText'].isna()==0]
dataset = dataset[dataset.reviewText != "None"]
print("before drop duplicates",dataset.shape)
dataset = dataset.drop_duplicates()
# dataset = dataset.drop_duplicates(subset=['overall', 'verified', 'reviewTime', 'reviewerID', 'asin',
#        'reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'vote',
#        'style', 'image'])
print("after drop duplicates",dataset.shape)

print(dataset)


dataset_reviews = dataset[['asin', 'reviewText']]

lemmatizer = WordNetLemmatizer()
stop_set = set(stopwords.words("english"))


def lemmatize(word):
    return lemmatizer.lemmatize(word)

def pre_processing(text):
	text_processed = text.translate(str.maketrans('', '', string.punctuation))
	text_processed = text_processed.split()
	result = []
	for word in text_processed:
		word_processed = word.lower()
		if word_processed not in stop_set:
			word_processed = lemmatize(word_processed)
			result.append(word_processed)
	return result


pandarallel.initialize(progress_bar=True)
dataset_reviews['processed_review'] = dataset_reviews['reviewText'].parallel_apply(pre_processing)
print(dataset_reviews['processed_review'])
dataset_reviews['processed_review'] = dataset_reviews['processed_review'].parallel_apply(lambda x: ' '.join(x))
print(dataset_reviews['processed_review'])
dataset_reviews['processed_review'] = dataset_reviews['processed_review'].replace('\d+', '', regex=True).str.replace(r'\b\w\b', '', regex=True)
print(dataset_reviews)


dataset_reviews['num_reviews'] = dataset_reviews.groupby('asin')['asin'].transform('count')
dataset_reviews_sub = dataset_reviews[dataset_reviews.num_reviews>1]
print(dataset_reviews_sub)


grouped = dataset_reviews_sub.groupby("asin")

# Initialize a dictionary to store average cosine similarity
average_similarity = {}

# Define a function to calculate average cosine similarity
def calculate_average_similarity(tfidf_matrix):
    similarity_matrix = cosine_similarity(tfidf_matrix)
    np.fill_diagonal(similarity_matrix, 0)
    avg_cosine_similarity = np.triu(similarity_matrix).mean()
    return avg_cosine_similarity

# Iterate through each "asin" group
for asin, group_data in grouped:
    # Calculate TF-IDF scores for the reviews in the group
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(group_data["processed_review"])
    
    # Calculate average cosine similarity
    avg_sim = calculate_average_similarity(tfidf_matrix)
    average_similarity[asin] = avg_sim
print("done")

similarity_df = pandas.DataFrame(list(average_similarity.items()), columns=["asin", "avg_sim_TF_IDF"])
print(similarity_df)


'''save the label_product_reviewer_pair variable'''
merge_HK_review_meta_inner_with_new_spike = pandas.read_csv("parsed_files/merge_HK_review_meta_inner_with_new_spike.csv", lineterminator='\n') 
merge_HK_review_meta_inner_with_new_spike.drop(['avg_sim_TF_IDF'], axis=1, inplace=True)
print(merge_HK_review_meta_inner_with_new_spike.columns)
print(merge_HK_review_meta_inner_with_new_spike.shape)

merge_HK_review_meta_inner_with_abFK_with_new_spike = pandas.read_csv("parsed_files/merge_HK_review_meta_inner_with_abFK_with_new_spike.csv", lineterminator='\n')
merge_HK_review_meta_inner_with_abFK_with_new_spike.drop(['avg_sim_TF_IDF'], axis=1, inplace=True)
print(merge_HK_review_meta_inner_with_abFK_with_new_spike.columns)
print(merge_HK_review_meta_inner_with_abFK_with_new_spike.shape)

merge_HK_review_meta_inner_with_new_spike = merge_HK_review_meta_inner_with_new_spike.merge(similarity_df, on=['asin'], suffixes=['_meta','_tf_idf'], how='left')
print(merge_HK_review_meta_inner_with_new_spike.columns)
print(merge_HK_review_meta_inner_with_new_spike.shape)

merge_HK_review_meta_inner_with_abFK_with_new_spike = merge_HK_review_meta_inner_with_abFK_with_new_spike.merge(similarity_df, on=['asin'], suffixes=['_meta','_tf_idf'], how='left')
print(merge_HK_review_meta_inner_with_abFK_with_new_spike.columns)
print(merge_HK_review_meta_inner_with_abFK_with_new_spike.shape)

'''save'''
merge_HK_review_meta_inner_with_new_spike.to_csv("parsed_files/merge_HK_review_meta_inner_with_new_spike.csv",index=False)
merge_HK_review_meta_inner_with_abFK_with_new_spike.to_csv("parsed_files/merge_HK_review_meta_inner_with_abFK_with_new_spike.csv",index=False)



