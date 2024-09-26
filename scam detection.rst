.. code:: ipython3

    import pandas as pd

.. code:: ipython3

    df = pd.read_csv("D:\presentation\dataset.csv")

.. code:: ipython3

    df = df[['text','label']]

.. code:: ipython3

    df.head()




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
          <th>text</th>
          <th>label</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>House Dem Aide: We Didn’t Even See Comey’s Let...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Ever get the feeling your life circles the rou...</td>
          <td>0</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Why the Truth Might Get You Fired October 29, ...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Videos 15 Civilians Killed In Single US Airstr...</td>
          <td>1</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Print \nAn Iranian woman has been sentenced to...</td>
          <td>1</td>
        </tr>
      </tbody>
    </table>
    </div>



.. code:: ipython3

    df = df.sample(n=7000)
    df.shape




.. parsed-literal::

    (7000, 2)



.. code:: ipython3

    df.isnull().sum()




.. parsed-literal::

    text     12
    label     0
    dtype: int64



.. code:: ipython3

    df.dropna(inplace=True)

.. code:: ipython3

    #Clean Text
    import re
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    
    def clean_text(text):
        # Convert text to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove links
        text = re.sub(r'http\S+', '', text)
        
        # Tokenize the text
        words = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words]
        
        # Initialize Porter Stemmer
        stemmer = PorterStemmer()
        
        # Perform stemming
        stemmed_words = [stemmer.stem(word) for word in filtered_words]
        
        # Join the stemmed words back into a single string
        cleaned_text = ' '.join(stemmed_words)
        
        return cleaned_text

.. code:: ipython3

    df['clean_text'] = df['text'].apply(lambda x: clean_text(x))

.. code:: ipython3

    df['label'].value_counts()




.. parsed-literal::

    label
    1    3515
    0    3473
    Name: count, dtype: int64



.. code:: ipython3

    #Train Models
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix, classification_report
    
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(df['clean_text'], df['label'], test_size=0.2, random_state=42)
    
    # TF-IDF vectorization
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

.. code:: ipython3

    # Random Forest classifier
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = rf_classifier.predict(X_test_tfidf)
    
    # Confusion matrix and classification report
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(class_report)


.. parsed-literal::

    Confusion Matrix:
    [[635  50]
     [ 97 616]]
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.87      0.93      0.90       685
               1       0.92      0.86      0.89       713
    
        accuracy                           0.89      1398
       macro avg       0.90      0.90      0.89      1398
    weighted avg       0.90      0.89      0.89      1398
    
    

.. code:: ipython3

    from sklearn.linear_model import LogisticRegression
    
    # Logistic Regression classifier
    lr_classifier = LogisticRegression(random_state=42)
    lr_classifier.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred_lr = lr_classifier.predict(X_test_tfidf)
    
    # Confusion matrix and classification report for logistic regression
    conf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    class_report_lr = classification_report(y_test, y_pred_lr)
    
    print("Logistic Regression - Confusion Matrix:")
    print(conf_matrix_lr)
    print("\nLogistic Regression - Classification Report:")
    print(class_report_lr)


.. parsed-literal::

    Logistic Regression - Confusion Matrix:
    [[628  57]
     [ 41 672]]
    
    Logistic Regression - Classification Report:
                  precision    recall  f1-score   support
    
               0       0.94      0.92      0.93       685
               1       0.92      0.94      0.93       713
    
        accuracy                           0.93      1398
       macro avg       0.93      0.93      0.93      1398
    weighted avg       0.93      0.93      0.93      1398
    
    

.. code:: ipython3

    from sklearn.svm import SVC
    
    # SVM classifier
    svm_classifier = SVC(kernel='linear', random_state=42)
    svm_classifier.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred_svm = svm_classifier.predict(X_test_tfidf)
    
    # Confusion matrix and classification report for SVM
    conf_matrix_svm = confusion_matrix(y_test, y_pred_svm)
    class_report_svm = classification_report(y_test, y_pred_svm)
    
    print("SVM - Confusion Matrix:")
    print(conf_matrix_svm)
    print("\nSVM - Classification Report:")
    print(class_report_svm)


.. parsed-literal::

    SVM - Confusion Matrix:
    [[648  37]
     [ 42 671]]
    
    SVM - Classification Report:
                  precision    recall  f1-score   support
    
               0       0.94      0.95      0.94       685
               1       0.95      0.94      0.94       713
    
        accuracy                           0.94      1398
       macro avg       0.94      0.94      0.94      1398
    weighted avg       0.94      0.94      0.94      1398
    
    

.. code:: ipython3

    #Detection System
    def predict_fake_or_real(text):
        # Clean the input text
        cleaned_text = clean_text(text)
        
        # Transform the cleaned text using the TF-IDF vectorizer
        text_tfidf = tfidf_vectorizer.transform([cleaned_text])
        
        # Use the trained classifier to predict
        prediction = svm_classifier.predict(text_tfidf)
        
        # Map prediction to label
        label = "fake" if prediction[0] == 1 else "real"
        
        return label
    
    # Example usage:
    input_text = 'Organizing for Action, the activist group that morphed from Barack Obama’s first presidential campaign, has partnered with the   Indivisible Project for “online trainings” on how to protest President Donald Trump’s agenda. [Last week, Breitbart News extensively reported that Indivisible leaders are openly associated with groups financed by billionaire George Soros.  Politico earlier this month profiled Indivisible in an article titled, “Inside the protest movement that has Republicans reeling. ”  The news agency not only left out the Soros links, but failed to note that the organizations cited in its article as helping to amplify Indivisible’s message are either financed directly by Soros or have close ties to groups funded by the billionaire, as Breitbart News documented. Organizing for Action (OFA) is a   community organizing project that sprung from Obama’s 2012 campaign organization, Organizing for America, becoming a nonprofit described by the Washington Post as “advocate[ing] for the president’s policies. ” In a recent Facebook post titled, “Take a deep breath. Then take action,” OFA called on constituents to lobby particularly hard between now and February 26, when lawmakers will be in their home districts. The post included a link to a guide released by Indivisible on how to organize against Trump. “Stay tuned for online trainings and invitations to calls with coalition partners like Indivisible Guide,” the OFA post states. Paul Sperry, writing at the New York Post, relates: The manual, published with OFA partner “Indivisible,” advises protesters to go into halls quietly so as not to raise alarms, and “grab seats at the front of the room but do not all sit together. ” Rather, spread out in pairs to make it seem like the whole room opposes the Republican host’s positions. “This will help reinforce the impression of broad consensus. ” It also urges them to ask “hostile” questions  —   while keeping “a firm hold on the mic”  —   and loudly boo the the GOP politician if he isn’t “giving you real answers. ” “Express your concern [to the event’s hosts] they are giving a platform to   authoritarianism, racism, and corruption,” it says.   …    “Even the safest [Republican] will be deeply alarmed by signs of organized opposition,” the document states, “because these actions create the impression that they’re not connected to their district and not listening to their constituents. ” Sperry reported OFA “plans to stage 400 rallies across 42 states this year to attack Trump and Republicans over ObamaCare’s repeal. ” Earlier this month, NBC News reported on OFA’s new actions and its partnership with Indivisible: OFA has hired 14 field organizers in states home to key senators as part of its campaign to defend Obama’s signature healthcare law. To run that campaign, the group hired Saumya Narechania  —   the former national field director at Enroll America, which worked to sign people up for Obamacare  —   and a deputy campaign manager.   …    OFA says more than 1, 800 people have applied to its Spring Community Engagement Fellowship, a   training program,   of whom have not previously been involved with OFA. And the group has teamed up with Indivisible, a buzzy newcomer to the progressive movement, to offer organizing training that began Thursday night with a video conference. A combined 25, 000 people have registered to participate in those trainings, OFA said. Indivisible’s DC branch was implicated in a scuffle last week that reportedly injured a    staffer for Rep. Dana Rohrabacher ( ) as well as reportedly knocking a    to the ground.  Protesters claimed they were only delivering Valentine’s Day cards. Indivisible is a part of a coalition of activist groups slated to hold a massive   Tax March in Washington and at least 60 other locations on April 15. Unreported by the news media is that most of the listed partners and support organizers of the march are openly financed by Soros or have close links to Soros financing, as Breitbart News documented last week. Meanwhile, earlier this month, Politico profiled Indivisible and reported that “conservatives” are “spreading unfounded rumors” that the group is “being driven by wealthy donors like George Soros. ” Politico, however, seemingly failed to do even the most minimal research on the Indivisible leaders cited in the news outlet’s own profile.  Some of those personalities are openly associated with groups financed by Soros. Politico further failed to note that the organizations cited in its article as helping to amplify Indivisible’s message are either financed directly by Soros or have close ties to groups funded by the billionaire. Citing Angel Padilla, a   of the group, Politico reported: Dubbed “Indivisible,” the group launched as a way for Padilla and a handful of fellow   aides to channel their   heartbreak into a manual for quashing President Donald Trump’s agenda. They drafted a   protest guide for activists, full of pointers on how to bird dog their members of Congress in the language of Capitol insiders. The manual has since been downloaded over one million times. Indivisible says on its website that over 4, 500 local groups across the nation have “signed up to resist the Trump agenda in nearly every congressional district in the country. ” The manual has been utilized to form the basis of a protest movement. The group’s website states: “What’s more, you all are putting the guide into action —  showing up en masse to congressional district offices and events, and flooding the congressional phone lines. You’re resisting —  and it’s working. ” Politico reported on “unfounded” rumors being spread about Soros’s involvement with Indivisible (emphasis added by this reporter): Its handful of senior leaders count about 100 contributors to their national organizing work but insist that all are working on a volunteer basis. They know conservatives are spreading unfounded rumors that their success is being driven by wealthy donors like George Soros, which they flatly deny. That paragraph was followed by the following quote from   Padilla (emphasis again added by this reporter): “It doesn’t matter who we take money from  —   we’re always going to get blamed as a Soros group, even if we don’t take money from Soros,” said Padilla, now an analyst with the National Immigration Law Center. “That’s one of the attacks and that’s fine. ” While “Indivisible” has yet to disclose its donors, Politico failed to inform readers that the National Immigration Law Center where the news outlet reported Padilla serves as an analyst is financed by Soros’s Open Society Foundations. The Center has received numerous Open Society grants earmarked for general support. Also unmentioned by Politico is that Padilla previously served as an immigration policy consultant at the radical National Council of La Raza. Soros is a major La Raza donor. Politico went on to detail how Indivisible has been aided by MoveOn. org and the ACLU.  The news website failed to tell readers that MoveOn. org and the ACLU are both financed by Soros, a relevant tidbit given Politico’s claim about “unfounded rumors” that Indivisibles’ success was being driven by Soros .  The news website reported: In addition, MoveOn. org and the Working Families Party joined with Indivisible for its first nationwide call on Jan. 22. Nearly 60, 000 people phoned in that day, according to Levin and MoveOn organizing director Victoria Kaplan. Indivisible estimates that its second national call, on the impact of Trump’s immigration order with assistance from the ACLU and Padilla’s group, drew 35, 000 people. Politico also missed that, according to its Twitter account, another organizer of the conference call with MoveOn. org was the International Refugee Assistance Project, a project of the Urban Justice Center, another recipient of an Open Society grant. Taryn Higashi, executive director of the Center’s International Refugee Assistance Project, currently serves on the Advisory Board of the International Migration Initiative of Soros’s Open Society Foundations. Politico further reported on Indivisible’s ties to the organizers of last month’s   Women’s March while failing to mention that Soros reportedly has ties to more than 50 “partners” of that march. Also, this journalist first reported on the march leaders’ own close associations with Soros. Regarding Indivisible and the Women’s March, Politico reported: Indivisible is also embracing collaboration with other major   protest outlets. Leaders of the group were in communication with Women’s March organizers before their main event on Jan. 21, and that partnership will become official when the March unveils the third in its series of 10 direct actions that attendees have been asked to pursue in their communities. Another Indivisible leader mentioned in the Politico article is Jeremy Haile. Not reported by Politico is that is Haile served as federal advocacy counsel for the Sentencing Project.  The Sentencing Project is reportedly financed by Soros’s Open Society Foundations, which has also hosted the Project to promote its cause. Aaron Klein is Breitbart’s Jerusalem bureau chief and senior investigative reporter. He is a New York Times bestselling author and hosts the popular weekend talk radio program, “Aaron Klein Investigative Radio. ” Follow him on Twitter @AaronKleinShow. Follow him on Facebook'
    prediction = predict_fake_or_real(input_text)
    print("Prediction:", prediction)


.. parsed-literal::

    Prediction: real
    

.. code:: ipython3

    # Example usage:
    input_text = 'Email \nThe alleged connection between Donald Trump and Russia — asserted by Hillary Clinton in both the second and third presidential debates as a dodge for the leaked e-mails and other documents which have hounded her candidacy — has dominated much of the speculative attention of both the mainstream media and social media for weeks. Is there anything to it? Is Trump, in essence, Putin’s puppet, as Clinton claimed in the third debate? \nIn an interesting twist of logic (if one can use that word to describe the incoherent and dissonant arguments made by a woman who still denies — despite mountains of evidence to the contrary — that she sent and received classified data over her unsecured, private e-mail server), Hillary Clinton at once claims that Russia is responsible for the hack of the servers of the Democratic National Committee (DNC) in late spring and that there is no way her server was hacked by anyone. Leveraging the anti-Russian sentiment to its fullest, she and others in the DNC have sought to shift the focus away from what was on the servers and toward blaming Russia for the hack. After all, if Russia is trying to throw the election to Trump, wouldn’t the patriotic vote be for Clinton? \nBy the time WikiLeaks announced that a trove of damning documents and e-mails would be published on the Internet, Clinton and the DNC (along with their lapdog media) were already in full-blown "it was Russia" mode. On the same day that a video of Donald Trump emerged in which he can be heard bragging of his sexual abuse of women — which he has since claimed was just “locker room talk” — WikiLeaks began a rolling release of the promised leaked documents. Both of these stories deserved media attention . Only one got it. \nBy the time the media was ready to report anything on the WikiLeaks dump, it was to speculate as to what part Russia played in the initial hack. Clinton — in her role as the staunch anti-Russian — used the timing of the second debate to attack Trump for the video and his supposed ties to Russia, which she plainly blamed for the leaked documents, saying to moderator Martha Raddatz: But, you know, let’s talk about what’s really going on here, Martha, because our intelligence community just came out and said in the last few days that the Kremlin, meaning Putin and the Russian government, are directing the attacks, the hacking on American accounts to influence our election. And WikiLeaks is part of that, as are other sites where the Russians hack information, we don’t even know if it’s accurate information, and then they put it out. We have never in the history of our country been in a situation where an adversary, a foreign power, is working so hard to influence the outcome of the election. And believe me, they’re not doing it to get me elected. They’re doing it to try to influence the election for Donald Trump. \nClinton went even further in the third debate , bringing up Putin in an attempt to dodge a question about a leaked document showing that she gave a speech to a foreign bank in which she said, “My dream is a hemispheric common market with open trade and open borders.” She — again — put the emphasis on the fact that WikiLeaks was providing the leaked documents rather than on what the leaked documents have to say. Then she claimed, on the authority of “17 of our intelligence agencies,” that Russia was behind the hacked and leaked documents: But you are very clearly quoting from WikiLeaks. And what’s really important about WikiLeaks is that the Russian government has engaged in espionage against Americans. They have hacked American websites, American accounts of private people, of institutions. Then they have given that information to WikiLeaks for the purpose of putting it on the Internet. This has come from the highest levels of the Russian government, clearly, from Putin himself, in an effort, as 17 of our intelligence agencies have confirmed, to influence our election. \nTrump rebutted by bringing the conversation back on topic (not his usual strongest point), saying: She wants open borders. People are going to pour into our country. People are going to come in from Syria. She wants 550 percent more people than Barack Obama, and he has thousands and thousands of people. They have no idea where they come from. And you see, we are going to stop radical Islamic terrorism in this country. She won’t even mention the words, and neither will President Obama. So I just want to tell you, she wants open borders. Now we can talk about Putin. I don’t know Putin. He said nice things about me. If we got along well, that would be good. If Russia and the United States got along well and went after ISIS, that would be good. He has no respect for her. He has no respect for our president. And I’ll tell you what: We’re in very serious trouble, because we have a country with tremendous numbers of nuclear warheads — 1,800, by the way — where they expanded and we didn’t, 1,800 nuclear warheads. And she’s playing chicken. Look, Putin... \nAt this point, Clinton interrupted to say, “Well, that’s because he’d rather have a puppet as president of the United States.” \nSo, getting past the he-said-she-said, what are the facts as we know them? \nOn Monday, the New York Times reported that after conducting a months-long “investigation into a Russian role in the American presidential campaign” in which “agents scrutinized advisers close to Donald J. Trump, looked for financial connections with Russian financial figures, searched for those involved in hacking the computers of Democrats, and even chased a lead — which they ultimately came to doubt — about a possible secret channel of email communication from the Trump Organization to a Russian bank,” the FBI “sees no clear link” between Trump and Russia. Of course, in typical leftist fashion, the Times not-so-gently insinuates that the FBI is guilty of pulling punches in the investigation. Interestingly, the Times seemed to miss that that is exactly what happened in the FBI\'s investigation into Clinton\'s e-mail server. \nAs for Clinton’s claims that the purpose of the hacked and leaked documents and e-mails was to influence the election for Trump, the Times article admits: Law enforcement officials say that none of the investigations so far have found any conclusive or direct link between Mr. Trump and the Russian government. And even the hacking into Democratic emails, F.B.I. and intelligence officials now believe, was aimed at disrupting the presidential election rather than electing Mr. Trump. \nAlso on Monday, Slate published an article claiming that a server belonging to Donald Trump was “communicating in a secretive fashion” with servers in Russia. By the next day, the Washington Post had debunked the Slate article, saying, “That secret Trump-Russia email server link is likely neither secret nor a Trump-Russia link.” Based in part on an interview with Naadir Jeewa, who “does consulting work on precisely the sorts of systems involved in” the scenario involving the server Slate claims was acting as a conduit between Trump and Russia, the article by the Post explains: To understand what\'s likely happening, we need to establish a few basics. First of all, the Trump server wasn\'t really a Trump server. It was much less of a Trump email server, for example, than Hillary Clinton\'s email server was hers. Clinton had a physical server that hosted her email. The trump-email.com domain that Alfa was connecting to was hosted by a company called Cendyn. Cendyn runs marketing systems for the hospitality industry, meaning that it offers an out-of-the-box solution for a company that owns a bunch of hotels to push out sales pitch emails to its customers. In other words, trump-email.com isn\'t the email server Trump used to send emails from his closet. It was a domain name that linked back to a Cendyn server. This is important for a few reasons. The first, Jeewa said, was that the trump-email.com was configured to reject a certain type of query from another server. Since its job was simply to push out thousands of enticements to come stay at Trump Soho (or whatever) it didn\'t need to receive many incoming requests (like incoming email). The second is that the conspiracy theory hinges on Trump\'s team using an offsite server hosted by someone else for its quiet communications with its Russian allies. Instead of, say, their own server, under their own control. Or an encrypted chat app. Or a phone call. \nThis writer has to admit to being amused to see the shoe on the other foot (or the tinfoil hat on the other head, in this instance) as the Left trots out wild conspiracy theories to create a Trump-Putin connection to make Clinton retroactively correct. In point of fact, Clinton might be better off trying to implicate Trump in the Kennedy assassination. \nBut, what about Clinton’s assertion that WikiLeaks is releasing documents pilfered by Russia for the sake of influencing the election for Trump? Aside from the FBI saying that there does not appear to be any attempt to influence the election for Trump, Julian Assange, the founder and public face of WikiLeaks, denies that Russian hackers are his source. \nIn an upcoming documentary made by Dartmouth Films, Assange speaks of the “Clinton camp” putting forth a “hysteria that Russia is responsible for everything.” He goes on to say, “Hillary Clinton has stated multiple times, falsely, that 17 US intelligence agencies had assessed that Russia was the source of our publications. That’s false — we can say that the Russian government is not the source.” \nFurthermore, the Kremlin denies Clinton’s claims. Russian presidential press secretary Dmitry Peskov called the claims “nonsense.” While that — in and of itself — would not carry much weight, when it is added to the body of evidence that includes Assange denying the Russian connection, the laughably far-fetched lengths to which the Left will got to promote easily debunkable conspiracy theories, the FBI stating that there is “no clear link” between Trump and Russia, and Trump himself denying that he has any dealings with Russia, it’s fairly easy to see that the only thing there is to see here is an imploding campaign by the woman who — the last time she occupied 1600 Pennsylvania Avenue — claimed that her husband’s impeachment was the result of a “vast right-wing conspiracy.” \nSome things never change. Photos: AP Images  '
    prediction = predict_fake_or_real(input_text)
    print("Prediction:", prediction)


.. parsed-literal::

    Prediction: fake
    

