from flask import Flask, render_template, request, redirect, url_for,session
import sqlite3
app = Flask(__name__)

#=============homepage modules start =====================
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')


@app.route('/register', methods=['POST', 'GET'])
def register():
    msg = ""
    if request.method == 'POST':
        try:
            name = request.form['name']
            loginid = request.form['loginid']
            email = request.form['email']
            password = request.form['password']
            branch = request.form['branch']
            collagename = request.form['collagename']
            phone = request.form['phone']
            locality = request.form['locality']
            state = request.form['state']                    
            
            status = 'waiting'             
            with sqlite3.connect('database.db') as con:
                cur = con.cursor()
                cur.execute("INSERT INTO students (name, loginid, email, password, branch, collagename, phone, locality, state, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (name, loginid, email, password, branch, collagename, phone, locality, state, status))
                con.commit()
                msg = "Record successfully added to database"
        except Exception as e:
            con.rollback()
            msg = f"Error in the INSERT: {e}"
        finally:
            con.close()
            return render_template('register.html', msg=msg)
    return render_template('register.html')



@app.route('/adminlogin',methods=['POST','GET'])
def adminlogin():
    error = None
    if request.method == 'POST':
        loginid = request.form['loginid']
        password = request.form['password']
        if loginid == 'admin' and password == 'admin':
            return render_template('admins/adminhome.html')            
        else:
            error = 'Invalid Credentials. Please try again.'
    return render_template('adminlogin.html',error=error)

@app.route('/userlogin', methods=['POST', 'GET'])
def userlogin():
    error = None
    if request.method == 'POST':
        loginid = request.form['loginid']
        password = request.form['password']        
        with sqlite3.connect('database.db') as con:
            con.row_factory = sqlite3.Row  
            cur = con.cursor()
            cur.execute("SELECT * FROM students WHERE name = ? AND password = ?", (loginid, password))
            user = cur.fetchone()            
            if user:
                if user['status'] == 'waiting':
                    error = "Your account is not activated yet. Please contact admin."
                else:                    
                    return render_template('users/userhome.html')                    
            else:
                error = "Invalid credentials. Please try again."
    return render_template('userlogin.html', error=error)

#=============homepage modules ending =====================

#===================admin panel start here ================= 


@app.route('/adminhome')
def adminhome():
    return render_template('admins/adminhome.html')


@app.route('/list')
def list():
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT rowid, * FROM students")
    rows = cur.fetchall()
    con.close()    
    return render_template('admins/userlist.html',rows=rows)


@app.route('/delete_user/<int:uid>', methods=['GET'])
def delete_user(uid):
    with sqlite3.connect('database.db') as con:
        cur = con.cursor()
        cur.execute("DELETE FROM students WHERE rowid = ?", (uid,))
        con.commit()
    return redirect(url_for('list'))


    
@app.route('/activate_user/<int:uid>')
def activate_user(uid):
    with sqlite3.connect('database.db') as con:
        cur = con.cursor()
        cur.execute("UPDATE students SET status = 'activated' WHERE rowid = ?", (uid,))
        con.commit()
    return redirect(url_for('list'))


@app.route('/edit_user/<int:uid>', methods=['GET', 'POST'])
def edit_user(uid):
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        status = request.form['status']        
        with sqlite3.connect('database.db') as con:
            cur = con.cursor()
            cur.execute("UPDATE students SET name = ?, email = ?, password = ?, status = ? WHERE rowid = ?", (name, email, password, status, uid))
            con.commit()
        return redirect(url_for('list'))

    else:
        with sqlite3.connect('database.db') as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()
            cur.execute("SELECT * FROM students WHERE rowid = ?", (uid,))
            user = cur.fetchone()
        return render_template('admins/edit_user.html', user=user, uid=uid)


#===================admin panel end =================
import pandas as pd
import numpy as np
import os
from flask import current_app
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import matplotlib.pyplot as plt
import seaborn as sns 

@app.route('/dataset')
def dataset():   
    path = os.path.join(current_app.root_path, 'media', 'adult.csv')    
    path1 = pd.read_csv(path,nrows=200)
    data = path1.to_html()    
    return render_template('users/dataset.html',data=data)


@app.route('/dataset1')
def dataset1():   
    path = os.path.join(current_app.root_path, 'media', 'adult_encoded.csv')    
    path1 = pd.read_csv(path,nrows=200)
    data = path1.to_html()    
    return render_template('users/dataset1.html',data=data)





# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


@app.route('/training')
def training():
    path = os.path.join(current_app.root_path, 'media', 'adult.csv')   
    
    df = pd.read_csv(path)
    print(df)
    print(df.shape)
    print(df.info)
    print(df.isna().sum())
    print(df['sex'].value_counts())
    print(df['income'].value_counts())
    print(df['education'].value_counts())
    print(df.describe())
        # Step 3: Preprocessing
    df.replace('?', pd.NA, inplace=True)
    df.dropna(inplace=True)
    print(df)
        # Step 4: Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le
    print(df)
        # Show dataset info
    print("Dataset Preview:\n", df.head())
    print("\nShape:", df.shape)

        # Select only numeric (integer/float) columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    # print("\nNumeric Columns:", list(numeric_cols))

        # ----------- BAR PLOTS -----------
    print("\nDrawing Bar Plots for numeric fields...")
    for col in numeric_cols[:3]:
        plt.figure(figsize=(8, 4))
        plt.bar(df.index[:20], df[col].head(20), color='cornflowerblue')
        plt.title(f'Bar Plot of {col} (first 20 rows)')
        plt.xlabel('Index')
        plt.ylabel(col)
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # ----------- PIE PLOTS -----------
    print("\nDrawing Pie Plots for numeric fields...")
    for col in numeric_cols[:3]:
        plt.figure(figsize=(6, 6))
        plt.pie(df[col].head(10), 
                labels=[f'Row{i}' for i in range(10)], 
                autopct='%1.1f%%', 
                startangle=90)
        plt.title(f'Pie Plot of {col} (first 10 rows)')
        plt.show()

        # Step 4.5: Correlation heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df.corr()

    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show() 
    # print("congrates danaiah")
        # Step 4.6: Bar plot of target class
    plt.figure(figsize=(6, 4))
    sns.countplot(x='income', data=df, palette='Set2')
    # Optional: Add labels
    plt.title('Income Class Distribution')
    plt.xlabel('Income Class (Encoded)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['<=50K', '>50K'])  # Adjust if your encoding is different
    plt.tight_layout()
    plt.show()

        # Step 5: Split features and target
    X = df.drop('income', axis=1)
    y = df['income'] 
    from imblearn.over_sampling import SMOTE

    # # Apply SMOTE after splitting features and target
    # smote = SMOTE(random_state=42)
    # X_balanced, y_balanced = smote.fit_resample(X, y)

    # # Check new class distribution
    # from collections import Counter
    # print("Balanced class distribution:", Counter(y_balanced))


    # Step 3: Apply SMOTE to balance the dataset
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    # Step 4: Combine balanced X and y into one DataFrame
    balanced_df = pd.concat([X_res, y_res], axis=1)
    print(balanced_df.value_counts())



# Step 4.6: Bar plot of target class
    plt.figure(figsize=(6, 4))
    sns.countplot(x='income', data=balanced_df, palette='Set2')
    # Optional: Add labels
    plt.title('Income Class Distribution')
    plt.xlabel('Income Class (Encoded)')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['<=50K', '>50K'])  # Adjust if your encoding is different
    plt.tight_layout()
    plt.show()

        # Step 6: Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Step 7: Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)  # No scaling needed for RF

   
    # Step 8: Predictions
    rf_pred = rf_model.predict(X_test)

    # Step 9: Evaluation
    from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

    acc = accuracy_score(y_test, rf_pred)
    cm = confusion_matrix(y_test, rf_pred)

    print("✅ Random Forest Accuracy:", acc)
    print("\nConfusion Matrix:\n", cm)

   
    import pickle
    # Save Random Forest model
    # joblib.dump(rf_model, 'random_forest_model.pkl') 

    with open('media/random_forest_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    return render_template('users/training.html',accuracy_score=acc)


from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from flask import request

from flask import Flask, render_template, request
import numpy as np
import os
       # Load the trained model from media folder
# model_path = os.path.join(app.root_path, 'media\random_forest_model.pkl')
# model_path = "random_forest_model.pkl"
# # model_path = "C:\\Users\\14s dq 2535\\Desktop\\Renu\\financial forecasting on adult\\media\\random_forest_model.pkl"
# with open(model_path, 'rb') as file:
#     model = pickle.load(file)


# @app.route('/predication', methods=['GET', 'POST'])
# def predication():
#     # import pickle
#     # model_path = os.path.join(app.root_path, 'media','random_forest_model_balanced.pkl')
#     # model_path = "random_forest_model.pkl"
#     # model_path = "C:\\Users\\14s dq 2535\\Desktop\\Renu\\financial forecasting on adult\\media\\random_forest_model.pkl"
#     from joblib import load

#     model_path = os.path.join(app.root_path, 'media', 'random_forest_model_balanced.pkl')
#     model = load(model_path)

#     # with open(model_path, 'rb') as file:
#     #     model = pickle.load(file)
#     if request.method == 'POST':
#         age = float(request.form.get("age"))
#         print(age)
#         workclass = float(request.form.get("workclass"))
#         print(workclass)
#         fnlwgt = float(request.form.get("fnlwgt"))
#         education = request.form.get("education")
#         education_num = request.form.get("education.num")
#         marital_status = request.form.get("marital.status")
#         occupation = request.form.get("occupation")
#         relationship = request.form.get("relationship")
#         race = request.form.get("race")
#         sex = request.form.get("sex")
#         capital_gain = request.form.get("capital.gain")
#         capital_loss = request.form.get("capital.loss")
#         hours_per_week = request.form.get("hours.per.week")
#         native_country = request.form.get("native.country")
#         encoded_input = [[age,workclass,fnlwgt,education,education_num,	marital_status,	occupation,	relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country]]
#         print(encoded_input) 
#         # encoded_input = [[74,	5,	88638,	10,	16,	4	,9,	2,	4,	0,	0,	3683,	20,	38]] #1----->second target class 
#         # Random Forest prediction (no scaling needed)
#         rf_prediction = model.predict(encoded_input)
#         # print("Random Forest Prediction:", rf_prediction[0])
#         if rf_prediction[0] == 0:
#             msg = "Adults Income Is = Less Then 50K"
#         else:
#             msg = "Adults Income Is = More Then 50K"
#         return render_template('users/predication.html',msg=msg)
#     return render_template('users/predication.html')
    
# ======================================
@app.route('/predication', methods=['GET', 'POST'])
def predication():
    from joblib import load
    import numpy as np
    import os

    model_path = os.path.join(app.root_path, 'media', 'random_forest_model_balanced.pkl')
    model = load(model_path)

    msg = None

    if request.method == 'POST':
        try:
            # Convert all inputs to float
            age = float(request.form.get("age"))
            workclass = float(request.form.get("workclass"))
            fnlwgt = float(request.form.get("fnlwgt"))
            education = float(request.form.get("education"))
            education_num = float(request.form.get("education_num"))
            marital_status = float(request.form.get("marital_status"))
            occupation = float(request.form.get("occupation"))
            relationship = float(request.form.get("relationship"))
            race = float(request.form.get("race"))
            sex = float(request.form.get("sex"))
            capital_gain = float(request.form.get("capital_gain"))
            capital_loss = float(request.form.get("capital_loss"))
            hours_per_week = float(request.form.get("hours_per_week"))
            native_country = float(request.form.get("native_country"))

            # Prepare input as numpy array
            encoded_input = np.array([[age, workclass, fnlwgt, education, education_num,
                                       marital_status, occupation, relationship, race, sex,
                                       capital_gain, capital_loss, hours_per_week, native_country]])

            print("✅ Encoded input:", encoded_input)

            # Prediction
            rf_prediction = model.predict(encoded_input)
            print("Prediction output:", rf_prediction)

            if rf_prediction[0] == 0:
                msg = "Less Than 50K"
            else:
                msg = "More Than 50K"

        except Exception as e:
            msg = f"⚠️ Error: {str(e)}"

    return render_template('users/predication.html', msg=msg)

# =====================================



#===================user panel start =================
@app.route('/userhome')
def userhome():
    return render_template('users/userhome.html')

#===================user panel start =================





if __name__ == '__main__':
    app.run(debug=True,port=8000)
