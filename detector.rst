.. code:: ipython3

    import numpy as np
    import pandas as pd
    import re
    from nltk.corpus import stopwords # the for of in with
    from nltk.stem.porter import PorterStemmer # loved loving == love
    from sklearn.feature_extraction.text import TfidfVectorizer # loved = [0.0]
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

.. code:: ipython3

    news_df = pd.read_csv("D:\presentation\dataset.csv")

.. code:: ipython3

    news_df.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id</th>
          <th>title</th>
          <th>author</th>
          <th>text</th>
          <th>label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
          <td>Darrell Lucus</td>
          <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
          <td>Daniel J. Flynn</td>
          <td>Ever get the feeling your life circles the rou...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>Why the Truth Might Get You Fired</td>
          <td>Consortiumnews.com</td>
          <td>Why the Truth Might Get You Fired October 29, ...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>15 Civilians Killed In Single US Airstrike Hav...</td>
          <td>Jessica Purkiss</td>
          <td>Videos 15 Civilians Killed In Single US Airstr...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>Iranian woman jailed for fictional unpublished...</td>
          <td>Howard Portnoy</td>
          <td>Print \nAn Iranian woman has been sentenced to...</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    news_df.shape




.. parsed-literal::

    (20800, 5)



.. code:: ipython3

    news_df.isna().sum()




.. parsed-literal::

    id           0
    title      558
    author    1957
    text        39
    label        0
    dtype: int64



.. code:: ipython3

    news_df = news_df.fillna(' ')

.. code:: ipython3

    news_df.isna().sum()




.. parsed-literal::

    id        0
    title     0
    author    0
    text      0
    label     0
    dtype: int64



.. code:: ipython3

    news_df['content'] = news_df['author']+" "+news_df['title']
    news_df




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>id</th>
          <th>title</th>
          <th>author</th>
          <th>text</th>
          <th>label</th>
          <th>content</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>0</td>
          <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
          <td>Darrell Lucus</td>
          <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
          <td>1</td>
          <td>Darrell Lucus House Dem Aide: We Didn’t Even S...</td>
        </tr>
        <tr>
          <th>1</th>
          <td>1</td>
          <td>FLYNN: Hillary Clinton, Big Woman on Campus - ...</td>
          <td>Daniel J. Flynn</td>
          <td>Ever get the feeling your life circles the rou...</td>
          <td>0</td>
          <td>Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...</td>
        </tr>
        <tr>
          <th>2</th>
          <td>2</td>
          <td>Why the Truth Might Get You Fired</td>
          <td>Consortiumnews.com</td>
          <td>Why the Truth Might Get You Fired October 29, ...</td>
          <td>1</td>
          <td>Consortiumnews.com Why the Truth Might Get You...</td>
        </tr>
        <tr>
          <th>3</th>
          <td>3</td>
          <td>15 Civilians Killed In Single US Airstrike Hav...</td>
          <td>Jessica Purkiss</td>
          <td>Videos 15 Civilians Killed In Single US Airstr...</td>
          <td>1</td>
          <td>Jessica Purkiss 15 Civilians Killed In Single ...</td>
        </tr>
        <tr>
          <th>4</th>
          <td>4</td>
          <td>Iranian woman jailed for fictional unpublished...</td>
          <td>Howard Portnoy</td>
          <td>Print \nAn Iranian woman has been sentenced to...</td>
          <td>1</td>
          <td>Howard Portnoy Iranian woman jailed for fictio...</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>20795</th>
          <td>20795</td>
          <td>Rapper T.I.: Trump a ’Poster Child For White S...</td>
          <td>Jerome Hudson</td>
          <td>Rapper T. I. unloaded on black celebrities who...</td>
          <td>0</td>
          <td>Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...</td>
        </tr>
        <tr>
          <th>20796</th>
          <td>20796</td>
          <td>N.F.L. Playoffs: Schedule, Matchups and Odds -...</td>
          <td>Benjamin Hoffman</td>
          <td>When the Green Bay Packers lost to the Washing...</td>
          <td>0</td>
          <td>Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...</td>
        </tr>
        <tr>
          <th>20797</th>
          <td>20797</td>
          <td>Macy’s Is Said to Receive Takeover Approach by...</td>
          <td>Michael J. de la Merced and Rachel Abrams</td>
          <td>The Macy’s of today grew from the union of sev...</td>
          <td>0</td>
          <td>Michael J. de la Merced and Rachel Abrams Macy...</td>
        </tr>
        <tr>
          <th>20798</th>
          <td>20798</td>
          <td>NATO, Russia To Hold Parallel Exercises In Bal...</td>
          <td>Alex Ansary</td>
          <td>NATO, Russia To Hold Parallel Exercises In Bal...</td>
          <td>1</td>
          <td>Alex Ansary NATO, Russia To Hold Parallel Exer...</td>
        </tr>
        <tr>
          <th>20799</th>
          <td>20799</td>
          <td>What Keeps the F-35 Alive</td>
          <td>David Swanson</td>
          <td>David Swanson is an author, activist, journa...</td>
          <td>1</td>
          <td>David Swanson What Keeps the F-35 Alive</td>
        </tr>
      </tbody>
    </table>
    <p>20800 rows × 6 columns</p>
    </div>



.. code:: ipython3

    news_df['content']




.. parsed-literal::

    0        Darrell Lucus House Dem Aide: We Didn’t Even S...
    1        Daniel J. Flynn FLYNN: Hillary Clinton, Big Wo...
    2        Consortiumnews.com Why the Truth Might Get You...
    3        Jessica Purkiss 15 Civilians Killed In Single ...
    4        Howard Portnoy Iranian woman jailed for fictio...
                                   ...                        
    20795    Jerome Hudson Rapper T.I.: Trump a ’Poster Chi...
    20796    Benjamin Hoffman N.F.L. Playoffs: Schedule, Ma...
    20797    Michael J. de la Merced and Rachel Abrams Macy...
    20798    Alex Ansary NATO, Russia To Hold Parallel Exer...
    20799              David Swanson What Keeps the F-35 Alive
    Name: content, Length: 20800, dtype: object



.. code:: ipython3

    #Stemming
    ps = PorterStemmer()
    def stemming(content):
        if not isinstance(content, str):
            content = str(content)
        stemmed_content = re.sub('[^a-zA-Z]',' ',content)
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        return stemmed_content

.. code:: ipython3

    news_df['content'] = news_df['content'].apply(stemming)
    news_df['content']




.. parsed-literal::

    0        darrel lucu hous dem aid even see comey letter...
    1        daniel j flynn flynn hillari clinton big woman...
    2                   consortiumnew com truth might get fire
    3        jessica purkiss civilian kill singl us airstri...
    4        howard portnoy iranian woman jail fiction unpu...
                                   ...                        
    20795    jerom hudson rapper trump poster child white s...
    20796    benjamin hoffman n f l playoff schedul matchup...
    20797    michael j de la merc rachel abram maci said re...
    20798    alex ansari nato russia hold parallel exercis ...
    20799                            david swanson keep f aliv
    Name: content, Length: 20800, dtype: object



.. code:: ipython3

    X = news_df['content'].values
    y = news_df['label'].values

.. code:: ipython3

    print(X)


.. parsed-literal::

    ['darrel lucu hous dem aid even see comey letter jason chaffetz tweet'
     'daniel j flynn flynn hillari clinton big woman campu breitbart'
     'consortiumnew com truth might get fire' ...
     'michael j de la merc rachel abram maci said receiv takeov approach hudson bay new york time'
     'alex ansari nato russia hold parallel exercis balkan'
     'david swanson keep f aliv']
    

.. code:: ipython3

    vector = TfidfVectorizer()
    vector.fit(X)
    X = vector.transform(X)

.. code:: ipython3

    print(X)


.. parsed-literal::

      (0, 15686)	0.28485063562728646
      (0, 13473)	0.2565896679337957
      (0, 8909)	0.3635963806326075
      (0, 8630)	0.29212514087043684
      (0, 7692)	0.24785219520671603
      (0, 7005)	0.21874169089359144
      (0, 4973)	0.233316966909351
      (0, 3792)	0.2705332480845492
      (0, 3600)	0.3598939188262559
      (0, 2959)	0.2468450128533713
      (0, 2483)	0.3676519686797209
      (0, 267)	0.27010124977708766
      (1, 16799)	0.30071745655510157
      (1, 6816)	0.1904660198296849
      (1, 5503)	0.7143299355715573
      (1, 3568)	0.26373768806048464
      (1, 2813)	0.19094574062359204
      (1, 2223)	0.3827320386859759
      (1, 1894)	0.15521974226349364
      (1, 1497)	0.2939891562094648
      (2, 15611)	0.41544962664721613
      (2, 9620)	0.49351492943649944
      (2, 5968)	0.3474613386728292
      (2, 5389)	0.3866530551182615
      (2, 3103)	0.46097489583229645
      :	:
      (20797, 13122)	0.2482526352197606
      (20797, 12344)	0.27263457663336677
      (20797, 12138)	0.24778257724396507
      (20797, 10306)	0.08038079000566466
      (20797, 9588)	0.174553480255222
      (20797, 9518)	0.2954204003420313
      (20797, 8988)	0.36160868928090795
      (20797, 8364)	0.22322585870464118
      (20797, 7042)	0.21799048897828688
      (20797, 3643)	0.21155500613623743
      (20797, 1287)	0.33538056804139865
      (20797, 699)	0.30685846079762347
      (20797, 43)	0.29710241860700626
      (20798, 13046)	0.22363267488270608
      (20798, 11052)	0.4460515589182236
      (20798, 10177)	0.3192496370187028
      (20798, 6889)	0.32496285694299426
      (20798, 5032)	0.4083701450239529
      (20798, 1125)	0.4460515589182236
      (20798, 588)	0.3112141524638974
      (20798, 350)	0.28446937819072576
      (20799, 14852)	0.5677577267055112
      (20799, 8036)	0.45983893273780013
      (20799, 3623)	0.37927626273066584
      (20799, 377)	0.5677577267055112
    

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=1)

.. code:: ipython3

    X_train.shape




.. parsed-literal::

    (16640, 17128)



.. code:: ipython3

    X_test.shape




.. parsed-literal::

    (4160, 17128)



.. code:: ipython3

    model = LogisticRegression()
    model.fit(X_train,y_train)




.. raw:: html

    <style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression()</pre></div></div></div></div></div>



.. code:: ipython3

    train_y_pred = model.predict(X_train)
    print("train accurracy :",accuracy_score(train_y_pred,y_train))


.. parsed-literal::

    train accurracy : 0.9868389423076923
    

.. code:: ipython3

    test_y_pred = model.predict(X_test)
    print("train accurracy :",accuracy_score(test_y_pred,y_test))


.. parsed-literal::

    train accurracy : 0.9766826923076923
    

.. code:: ipython3

    # prediction system
    
    input_data = X_test[10]
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        print('Fake news')
    else:
        print('Real news')


.. parsed-literal::

    Real news
    

.. code:: ipython3

    news_df['content'][10]




.. parsed-literal::

    'aaron klein obama organ action partner soro link indivis disrupt trump agenda'



