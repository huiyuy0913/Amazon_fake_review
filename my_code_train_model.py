import pandas as pd
import matplotlib.pyplot as plt
import operator

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

df = pd.read_csv('product_level_data_without_img_feats.csv.gz')

review_features = ['avg_review_rating',
		     'tfidf_review_body',
                   'avg_days_between_reviews', 'stdev_days_between_reviews',
                   'max_days_between_reviews', 'min_days_between_reviews', 
                   'share_helpful_reviews', 'share_1star', 'share_5star', 'share_photo', 'std_review_len']
network_features = ['pagerank', 'w_degree', 'clustering_coef', 'eigenvector_cent']
image_sim_features = ['min_sim', 'max_sim', 'mean_sim', 'std_sim', 'min_sim_review', 'max_sim_review',
       'mean_sim_review', 'std_sim_review', 'min_sim_product',
       'max_sim_product', 'mean_sim_product', 'std_sim_product']


############################## FUNCTIONS ##############################
def model_building(X_train, y_train, X_test, y_test, model):

	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)
	cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
	probs = model.predict_proba(X_test)[:,1]

	# print(cm)
	print("AUC, Accuracy, TN, TP, F1 Score")
	print("{}, {}, {}, {}, {}".format(metrics.roc_auc_score(y_test, model.predict_proba(X_test)[:,1]),
															  sum(cm.diagonal()) / X_test.shape[0],
															  cm[0,0] / sum(cm[0,:]),
															  cm[1,1] / sum(cm[1,:]),
															  metrics.f1_score(y_test, y_pred, average='weighted')))

	return probs

def classification_results(df, features=None, stars=None):

	if features == None:
		X = df.drop(['product_ID','fake'], axis=1)
		features = list(X.columns)
		y = df['fake']
	else:
		X = df[features]
		y = df['fake']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
	print(X_train.shape, X_test.shape)

	scaler = StandardScaler()
	X_train = scaler.fit_transform(X_train)
	X_test = scaler.transform(X_test)

	print("="*10 + "Logistic Regression" + "="*10)
	model = LogisticRegression(max_iter=400)
	model_building(X_train, y_train, X_test, y_test, model)

	print("="*10 + "Random Forest" + "="*10)
	model = RandomForestClassifier(random_state=42, 
	                               n_estimators=100,
	                               min_samples_leaf=3,
	                               min_samples_split=6,
	                               max_features='auto',
	                               max_depth=40,
	                               bootstrap=True,
	                               n_jobs=-1)
	model_building(X_train, y_train, X_test, y_test, model)

	print("="*10 + "RF Feature Importance" + "="*10)
	imps = model.feature_importances_
	feat_imp = {features[i]: imps[i] for i in range(len(features))}

	name = pd.DataFrame(X_test, columns=review_features+network_features)
	importances = pd.DataFrame({'feature': review_features+network_features, 'importance': model.feature_importances_})
	importances = importances.sort_values('importance', ascending=False)
	feature_mapping = {
                'clustering_coef': 'clustering coefficient', 'share_photo': 'share photo',
                'share_5star': 'share 5 star', 'eigenvector_cent': 'eigenvector cent',
                'max_days_between_reviews': 'max days between reviews', 'avg_days_between_reviews': 'avg days between reviews', 'avg_review_rating': 'avg review rating',
                'pagerank': 'pagerank', 'w_degree': 'degree', 'std_review_len': 'stdev review len', 'share_helpful_reviews': 'share helpful reviews',
                'share_1star': 'share 1 star', 'min_days_between_reviews': 'min days between reviews', 
                'stdev_days_between_reviews': 'stdev days between reviews',
				'tfidf_review_body': 'tfidf sim'
                                          
	}
	importances['feature'] = importances['feature'].map(feature_mapping)
	import seaborn as sns
	plt.figure(figsize=(3, 5))
	# sns.set(style="whitegrid")
	sns.barplot(x='importance', y='feature', data=importances, color='darkblue')
	plt.xlabel("Relative Importance")
	plt.ylabel("")
	# plt.title("Feature Importance Plot")
	plt.savefig("modified_relative_importances_with_tf_idf.png", bbox_inches='tight')
	plt.show()

	if len(features) > 100:
		print(sorted(feat_imp.items(), key=operator.itemgetter(1), reverse=True)[:50])
	else:
		print(sorted(feat_imp.items(), key=operator.itemgetter(1), reverse=True))

	print("="*10 + "SVC Linear" + "="*10)
	model = SVC(kernel='linear', probability=True)
	model_building(X_train, y_train, X_test, y_test, model)

	print("="*10 + "XGBoost" + "="*10)
	model = xgb.XGBClassifier()
	model_building(X_train, y_train, X_test, y_test, model)

	return

####################### RESULTS ##############################
print("\n+++++++++++++++++ All Features ++++++++++++++++\n")
classification_results(df, review_features+network_features)

