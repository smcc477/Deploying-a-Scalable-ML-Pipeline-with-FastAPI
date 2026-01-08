Model Details



This model is a supervised binary classification model trained to predict whether someone's income is greater than $50K based on census data. It uses a logistic regression classifier implemented with scikit-learn and was was trained as part of a machine learning DevOps pipeline.  It is only intended for educational purposes.



Intended Use



The intended use of this model is to demonstrate a machine learning workflow, including data preprocessing, model training, evaluation, deployment, and monitoring. It is not for supporting real-world decisions about individuals, employment, or financial eligibility.



Training Data



The model was trained using the U.S. Census Income dataset. The dataset contains demographic and employment-related attributes like, workclass, education, occupation, age, marital status, race, sex, and country. The target variable is a binary flag indicating whether an individual earns more than $50K.



The data was split into training and test sets using an 80/20 split with stratification on the target label.



Evaluation Data



The evaluation data consists of the 'held out' 20% split from the original data. This data was not used during training and was processed using the same preprocessing pipeline as the training data.



Metrics



The model was evaluated using precision, recall, and F1 score. These metrics were chosen to assess classification performance, in the presence of class imbalance.



The model achieved the following performance on the test:



Precision: 0.7191



Recall: 0.2742



F1 Score: 0.3970



Model performance was also evaluated on slices of the data for each category. Metrics for each slice were computed and saved to slice\_output.txt.



Ethical Considerations



The dataset used to train this model contains sensitive demographic attributes. Using demographic features in predictive models can introduce and even amplify bias. This model may produce different performance outcomes across different demographic groups, as demonstrated by the slice based results.



The model should NOT be used in production systems where decisions can negatively impact individuals.



Caveats and Recommendations



The model shows fairly low recall, indicating that it fails to identify a many of the positive cases. Performance isn't consistent across demographic slices, especially for categories with small sample sizes.



Future improvements could include experimenting with different models, modifying weights on classes, tuning hyperparameters , or maybe adding additional evaluation metrics. This model is intended solely for instructional purposes.

