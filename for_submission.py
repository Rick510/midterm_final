# using model
import joblib
model_pretrained=joblib.load('Classification.pk1')
import pandas as pd
df_test = pd.read_csv("C:\\Users\\rick\\AI code\\midterm\\test.csv")

prediction_final = model_pretrained.predict(df_test)


# submit file
forSubmissionDF=pd.DataFrame(columns=['id','target'])
forSubmissionDF['id']=range(414,690)
forSubmissionDF['target'] = prediction_final

forSubmissionDF.to_csv('for_submission_20230427.csv', index=False)

