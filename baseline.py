import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import sklearn.preprocessing as preprocessing
scaler = preprocessing.StandardScaler()

ad_data = pd.read_csv("%s\\ad.csv" % pre, ',', nrows=300000)
ad_data.describe() # 获取信息 see also:ad_data.info()
ad_data = ad_data.filter(regex='creativeID|adID|camgaignID|advertiserID|appPlatform') # 正则
user_data = pd.read_csv("%s\\user.csv" % pre, ',', nrows = 300000 )
train_data_re = train_data.filter(regex='label|clickTime|positionID|connectionType|telecomsOperator')
data_train = train_data_re.join(user_data) #user_data is used for both train and test 
data_train_fin = data_train.sample(frac=0.6)   # 32位python作孽啊

tf = data_train_fin
tf = pd.concat([data_train_fin])
clickTime_scale_param = scaler.fit(tf['clickTime'])
tf['clickTime_scaled'] = scaler.fit_transform(tf['clickTime'], clickTime_scale_param)
age_scale_param = scaler.fit(tf['age'])
tf['age_scaled'] = scaler.fit_transform(tf['age'], age_scale_param)
userID_scale_param = scaler.fit(tf['userID'])
tf['userID_scaled'] = scaler.fit_transform(tf['userID'], userID_scale_param)
positionID_scale_param = scaler.fit(tf['positionID'])
tf['positionID_scaled'] = scaler.fit_transform(tf['positionID'], positionID_scale_param)
hometown_scale_param = scaler.fit(tf['hometown'])
tf['hometown_scaled'] = scaler.fit_transform(tf['hometown'], hometown_scale_param)
residence_scale_param = scaler.fit(tf['residence'])
tf['residence_scaled'] = scaler.fit_transform(tf['residence'], hometown_scale_param)
tf = pd.DataFrame(tf)   #  scaling 数据变动大的（不收敛&memory error）
final_train_data = tf.filter(regex='label|connectionType|telecomsOperator|gender|education|marriageStatus|haveBaby|clickTimescaled|age_scaled|userId_scaled|positionID_scaled|hometown_scaled|residence_scaled')
final_train_data.describe()


# 利用 get_dummies 进行 one-hot, see also: onehotencoding
connectionType = pd.get_dummies(final_train_data['connectionType'], prefix='connectionType')
telecomsOperator = pd.get_dummies(final_train_data['telecomsOperator'], prefix='telecomsOperator')
gender = pd.get_dummies(final_train_data['gender'], prefix='gender')
education = pd.get_dummies(final_train_data['education'], prefix='education')
marriageStatus = pd.get_dummies(final_train_data['marriageStatus'], prefix='marriageStatus')
final_train_data = final_train_data.join(connectionType).join(telecomsOperator).join(gender).join(education).join(marriageStatus)

data = final_train_data
data = data.filter(regex='label|telecomsOperator_.|connectionType_.|gender_.|education_.|marriageStatus_.|haveB.|age_.|positionID.|home.|residence_')

final_train_data = data  # 最终训练数据
train_np = final_train_data.as_matrix()
survial = train_np[:, 0]
vari = train_np[:, 1:]
logistic = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
logistic.fit(vari, survial) # LR训练 说实话 数据总量不到6%，训练模型也是最初等的,也是心累


# 获取&预处理test数据
pre = "C:\\Users\\dell\\Desktop\\tencent_model\\pre"
data_test = pd.read_csv("%s\\test.csv" % pre, ',', nrows=None)
data_test = data_test.filter(regex='clickTime|positionID|connectionType|telecomsOperator')
user_test = pd.read_csv("%s\\user.csv" % pre,',', nrows=None)
final_data_test = data_test.join(user_test)

scaler = preprocessing.StandardScaler()
df = pd.concat([final_data_test])
clickTime_scale_param = scaler.fit(df['clickTime'])
df['clickTime_scaled'] = scaler.fit_transform(df['clickTime'], clickTime_scale_param)
age_scale_param = scaler.fit(df['age'])
df['age_scaled'] = scaler.fit_transform(df['age'], age_scale_param)
userID_scale_param = scaler.fit(df['userID'])
df['userID_scaled'] = scaler.fit_transform(df['userID'], userID_scale_param)
positionID_scale_param = scaler.fit(df['positionID'])
df['positionID_scaled'] = scaler.fit_transform(df['positionID'], positionID_scale_param)
hometown_scale_param = scaler.fit(df['hometown'])
df['hometown_scaled'] = scaler.fit_transform(df['hometown'], hometown_scale_param)
residence_scale_param = scaler.fit(df['residence'])
df['residence_scaled'] = scaler.fit_transform(df['residence'], hometown_scale_param)
df = pd.DataFrame(df)
final_test_data = df.filter(regex='label|connectionType|telecomsOperator|gender|education|marriageStatus|haveBaby|clickTimescaled|age_scaled|userId_scaled|positionID_scaled|hometown_scaled|residence_scaled')
connectionType = pd.get_dummies(final_test_data['connectionType'], prefix='connectionType')
telecomsOperator = pd.get_dummies(final_test_data['telecomsOperator'], prefix='telecomsOperator')
gender = pd.get_dummies(final_test_data['gender'], prefix='gender')
education = pd.get_dummies(final_test_data['education'], prefix='education')
marriageStatus = pd.get_dummies(final_test_data['marriageStatus'], prefix='marriageStatus')
final_test_data = final_test_data.join(connectionType).join(telecomsOperator).join(gender).join(education).join(marriageStatus)
final_test_data = final_test_data.filter(regex='telecomsOperator_.|connectionType_.|gender_.|education_.|marriageStatus_.|haveB.|age_.|positionID.|home.|residence_')

#预测probability
predic = logistic.predict_proba(final_test_data)[:, 1] 
predic # LR

# 写入csv并压缩
import zipfile
import os
frame = pd.DataFrame({'instanceID':data_test['instanceID'].values, 'prob':predic})
frame.sort_values("instanceID", inplace=True)
os.chdir("C:\\Users\\dell\\Desktop\\tencent_model")
frame.to_csv('submission.csv', index=False)
with zipfile.ZipFile("submission.zip", "w") as fout:
    fout.write("submission.csv", compress_type=zipfile.ZIP_DEFLATED) #添加到压缩文档中
    fout.close()
# 名次虽然靠后，也算是学到一点东西吧，待我换掉机子，换掉32位python再战
