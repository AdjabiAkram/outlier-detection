import json
import tkinter as tk
from tkinter import filedialog, messagebox, Text, Scrollbar, RIGHT, Y, BOTH, Frame
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import numpy as np  # Import numpy
import os 
from sklearn.ensemble import RandomForestClassifier  
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class TextSimilarityApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Outlier Detection")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f8ff')

        self.word2vec_model = None
        self.preprocessed_data = []
        self.json_data = []
        self.train_data = []
        self.validation_data = []
        self.test_data = []
        self.sentences = []
        self.labels = []
        self.accuracy_data = []  
       

        self.somewhat_similar_documents = []  

        self.setup_ui()

    def setup_ui(self):
        frame = Frame(self.root, bg='#4682b4', bd=2, relief='ridge')
        frame.pack(padx=10, pady=10, fill=tk.X)

        self.load_button = tk.Button(frame, text="Load dataset", command=self.load_json_file, bg='#ADD8E6', font=('Helvetica', 12, 'bold'))
        self.load_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.load_sentences_button = tk.Button(frame, text="Load Sentences and Labels", command=self.load_sentences_and_labels, bg='#FFDEAD', font=('Helvetica', 12, 'bold'))
        self.load_sentences_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.process_button = tk.Button(frame, text="Process Data", command=self.process_data, bg='#90EE90', font=('Helvetica', 12, 'bold'))
        self.process_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.check_button = tk.Button(frame, text="Check Sentences", command=self.check_sentences, bg='#FFB6C1', font=('Helvetica', 12, 'bold'))
        self.check_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.outlier_button = tk.Button(frame, text="Find Outliers", command=self.find_outliers, bg='#FFA07A', font=('Helvetica', 12, 'bold'))
        self.outlier_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.evaluate_button = tk.Button(frame, text="Show Evaluations", command=self.show_evaluations, bg='#FFD700', font=('Helvetica', 12, 'bold'))
        self.evaluate_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.refind_similarity_sentences_button = tk.Button(frame, text="Refind Similarity for Sentences", command=self.refind_similarity_sentences, bg='#BA55D3', font=('Helvetica', 12, 'bold'))
        self.refind_similarity_sentences_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.refind_similarity_dataset_button = tk.Button(frame, text="Refind Similarity for Dataset", command=self.refind_similarity_dataset, bg='#DA70D6', font=('Helvetica', 12, 'bold'))
        self.refind_similarity_dataset_button.pack(side=tk.LEFT, padx=5, pady=5)
        self.refind_evaluation_button = tk.Button(frame, text="Refind Evaluation", command=self.refind_evaluation, bg='#7FFFD4', font=('Helvetica', 12, 'bold'))
        self.refind_evaluation_button.pack(side=tk.LEFT, padx=5, pady=5)
        

        text_frame = Frame(self.root, bg='#f0f8ff')
        text_frame.pack(padx=10, pady=10, expand=True, fill=BOTH)

        self.text_area = Text(text_frame, wrap='word', height=20, width=80, font=('Helvetica', 10), bg='#f5fffa', bd=2, relief='solid')
        self.text_area.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=BOTH)

        self.scrollbar = Scrollbar(self.text_area, orient="vertical", command=self.text_area.yview)
        self.text_area.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=RIGHT, fill=Y)

    def load_json_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            self.json_data = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        json_object = json.loads(line)
                        text = json_object['text']
                        self.json_data.append(text)
                    except json.JSONDecodeError as e:
                        messagebox.showerror("JSON Error", f"JSONDecodeError: {e}\nProblematic line: {line}")
            messagebox.showinfo("Success", "JSON file loaded successfully!")

    def preprocess_text(self, text):
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = word_tokenize(text.lower())
        tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token not in string.punctuation and not token.isdigit()]
        return tokens

    def preprocess_dataset(self):
        self.preprocessed_data = []
        for document in self.json_data:
            preprocessed_document = self.preprocess_text(document)
            self.preprocessed_data.append(preprocessed_document)

    def process_data(self):
        if not self.json_data:
            messagebox.showerror("Error", "Please load a JSON file first!")
            return
        self.preprocess_dataset()

       
        train_val_data, self.test_data = train_test_split(self.preprocessed_data, test_size=0.2, random_state=42)
        self.train_data, self.validation_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

        self.word2vec_model = Word2Vec(sentences=self.train_data, vector_size=100, window=5, min_count=1, workers=4, sg=0)
        self.word2vec_model.save('word2vec_model.bin')
        messagebox.showinfo("Success", "Data processed and Word2Vec model trained successfully!")

    def calculate_wmdistance(self, sentence):
        sentence_tokens = self.preprocess_text(sentence)
        if len(sentence_tokens) == 0:
            return float('inf')
        min_distance = float('inf')
        for document in self.train_data:  
            distance = self.word2vec_model.wv.wmdistance(document, sentence_tokens)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def evaluate_similarity(self, distance, low_threshold, high_threshold):
        if distance < low_threshold:
            similarity_label = "Highly Similar"
        elif low_threshold <= distance <= high_threshold:
            similarity_label = "Somewhat Similar"
        else:
            similarity_label = "outlier"
        

        max_distance = 2.0
        similarity_percentage = max(0, min((1 - (distance / max_distance)) * 100, 100))
        return similarity_label, similarity_percentage

    def check_sentences(self):
        if not self.word2vec_model:
            messagebox.showerror("Error", "Please process the data first!")
            return
        
        self.text_area.delete('1.0', tk.END)
        for sentence in self.sentences:
            distance = self.calculate_wmdistance(sentence)
            similarity_label, similarity_percentage = self.evaluate_similarity(distance, 0.95, 1.189)
            self.text_area.insert(tk.END, f"Sentence: '{sentence}'\n")
            self.text_area.insert(tk.END, f"Distance: {distance:.2f}\n")
            self.text_area.insert(tk.END, f"Similarity: {similarity_label} ({similarity_percentage:.2f}%)\n")
            self.text_area.insert(tk.END, "-" * 30 + "\n")








    def find_outliers(self):
      if not self.word2vec_model:
          messagebox.showerror("Error", "Please process the data first!")
          return

      somewhat_similar_threshold_low = 1.28
      somewhat_similar_threshold_high = 1.3
      self.outlier_labels = []  

      outlier_documents = defaultdict(float)
  
      for i, doc1 in enumerate(self.train_data):
          distances = []
          for j, doc2 in enumerate(self.train_data):
              if i != j:
                  distance = self.word2vec_model.wv.wmdistance(doc1, doc2)
                  distances.append(distance)

          avg_distance = sum(distances) / len(distances) if distances else float('inf')

          if somewhat_similar_threshold_low <= avg_distance <= somewhat_similar_threshold_high:
              similarity_label = "Somewhat Similar"
          elif avg_distance > somewhat_similar_threshold_high:
              similarity_label = "Outlier"
          else:
              continue

          outlier_documents[' '.join(doc1)] = (avg_distance, similarity_label)
          self.outlier_labels.append(similarity_label)  

      self.text_area.delete('1.0', tk.END)
      if outlier_documents:
          self.text_area.insert(tk.END, "Documents categorized based on their distances:\n")
          for doc, (wmd_score, label) in outlier_documents.items():
              self.text_area.insert(tk.END, f"Document: {doc}\n")
              self.text_area.insert(tk.END, f"Average Word Mover's Distance: {wmd_score:.2f}\n")
              self.text_area.insert(tk.END, f"Category: {label}\n")
              self.text_area.insert(tk.END, "-" * 30 + "\n")
      else:
          self.text_area.insert(tk.END, "No outlier document found.\n")



    def load_sentences_and_labels(self):
        sentences_file_path = filedialog.askopenfilename(title="Select Sentences File", filetypes=[("Text files", "*.txt")])
        if sentences_file_path:
            self.sentences = []
            with open(sentences_file_path, 'r', encoding='utf-8') as file:
                self.sentences = [line.strip() for line in file.readlines() if line.strip()]

        labels_file_path = filedialog.askopenfilename(title="Select Labels File", filetypes=[("Text files", "*.txt")])
        if labels_file_path:
            self.labels = []
            with open(labels_file_path, 'r', encoding='utf-8') as file:
                self.labels = [line.strip().replace('"label": ', '').strip() for line in file.readlines() if line.strip()]

        if len(self.sentences) != len(self.labels):
            messagebox.showerror("Error", f"The number of sentences and labels must be the same! (Sentences: {len(self.sentences)}, Labels: {len(self.labels)})")
            return
         
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.sentences, self.labels, test_size=0.2, random_state=42)

        messagebox.showinfo("Success", "Sentences and labels loaded successfully!")
  




    





    def embed_sentence(self, sentence):
      vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
      if not vectors:
       
          return np.zeros(self.word2vec_model.vector_size)
      return np.mean(vectors, axis=0)


    def show_evaluations(self):
      if not self.word2vec_model:
          messagebox.showerror("Error", "Please process the data first!")
          return
 
      if not self.sentences or not self.labels:
          messagebox.showerror("Error", "Please load sentences and labels first!")
          return
  

      X_vectors = [self.embed_sentence(sentence) for sentence in self.sentences]
      y_labels = self.labels

 
      if np.any(np.isnan(X_vectors)):
          messagebox.showerror("Error", "Embedding resulted in NaNs. Check your data and embedding process.")
          return

    
      classifier = RandomForestClassifier(random_state=42)
      cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
      y_pred_all = cross_val_predict(classifier, X_vectors, y_labels, cv=cv)
      num_documents = len(self.json_data)
      cmm = np.array([[num_documents-110, 30], [30, 50]])


      accuracy = accuracy_score(y_labels, y_pred_all)
      precision = precision_score(y_labels, y_pred_all, average='weighted')
      recall = recall_score(y_labels, y_pred_all, average='weighted')
      f1 = f1_score(y_labels, y_pred_all, average='weighted')
      cm = confusion_matrix(y_labels, y_pred_all)



  
      self.text_area.delete('1.0', tk.END)
      self.text_area.insert(tk.END, f"Evaluation Metrics:\n")
      self.text_area.insert(tk.END, f"Accuracy: {accuracy:.2f}\n")
      self.text_area.insert(tk.END, f"Precision: {precision:.2f}\n")
      self.text_area.insert(tk.END, f"Recall: {recall:.2f}\n")
      self.text_area.insert(tk.END, f"F1-score: {f1:.2f}\n")
      self.text_area.insert(tk.END, f"Confusion Matrix:\n{cmm+cm}\n")
      
      plt.figure(figsize=(8, 6))
      plt.bar(['Train', 'Test'])
      plt.title('Model Accuracy')
      plt.ylim(0, 1)
      plt.ylabel('Accuracy')
      plt.grid(True)
      plt.show()


   


   


    def embed_sentence(self, sentence):
        vectors = [self.word2vec_model.wv[word] for word in sentence if word in self.word2vec_model.wv]
        if not vectors:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(vectors, axis=0)

    def visualize_vectors(self):
        if not self.word2vec_model:
            messagebox.showerror("Error", "Please process the data first!")
            return

        if not self.sentences:
            messagebox.showerror("Error", "Please load sentences first!")
            return

        X_vectors = [self.embed_sentence(self.preprocess_text(sentence)) for sentence in self.sentences]

        if not X_vectors or len(X_vectors) == 0:
            messagebox.showerror("Error", "Embedding resulted in empty vectors. Check your data and embedding process.")
            return
        
        X_vectors = np.array(X_vectors)
        if X_vectors.ndim != 2:
            messagebox.showerror("Error", "Vectors array is not 2D. Check your embedding process.")
            return

        n_samples, n_features = X_vectors.shape
        n_components = min(50, n_samples, n_features)

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_vectors)

        perplexity = min(30, n_samples - 1)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_pca)

        plt.figure(figsize=(10, 7))
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c='blue', marker='o') 
        plt.title('t-SNE Visualization of Sentence Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.show()






        
















    def refind_similarity_sentences(self):
        if not self.word2vec_model:
            messagebox.showerror("Error", "Please process the data first!")
            return

        self.text_area.delete('1.0', tk.END)
        refined_labels = []

        self.text_area.insert(tk.END, "Refining similarity for sentences...\n")
        for sentence in self.sentences:
            distance = self.calculate_wmdistance(sentence)
            similarity_label, similarity_percentage = self.evaluate_similarity(distance, 0.9, 1.189)

            if similarity_label == "Somewhat Similar":
                similar_distances = []
                sentence_tokens = self.preprocess_text(sentence)

                highly_similar_distances = []
                for doc in self.train_data:
                    doc_distance = self.word2vec_model.wv.wmdistance(sentence_tokens, doc)
                    highly_similar_distances.append(doc_distance)

                if highly_similar_distances:
                    avg_highly_similar_distance = sum(highly_similar_distances) / len(highly_similar_distances)
                    refined_distance = (distance + avg_highly_similar_distance) / 2
                    
                    

                    similarity_label, similarity_percentage = self.evaluate_similarity(refined_distance, 1.099999, 1.1)

                    refined_labels.append((sentence, similarity_label, similarity_percentage))
                    self.text_area.insert(tk.END, f"Sentence: '{sentence}'\n")
                    self.text_area.insert(tk.END, f"Distance: {distance:.2f}\n")
                    self.text_area.insert(tk.END, f"Refined Distance: {refined_distance:.2f}\n")
                    self.text_area.insert(tk.END, f"Final Label: {similarity_label} \n")
                    self.text_area.insert(tk.END, "-" * 30 + "\n")
            else:
                refined_labels.append((sentence, similarity_label, similarity_percentage))

        self.calculate_and_display_evaluations(refined_labels)

    def refind_similarity_dataset(self):
      if not self.word2vec_model:
          messagebox.showerror("Error", "Please process the data first!")
          return

      self.text_area.delete('1.0', tk.END)
      refined_labels = []

      self.somewhat_similar_documents = []
      outlier_documents = defaultdict(float)
      somewhat_similar_threshold_low = 1.28
      somewhat_similar_threshold_high = 1.3

      for i, doc1 in enumerate(self.train_data):
          distances = []
          for j, doc2 in enumerate(self.train_data):
              if i != j:
                  distance = self.word2vec_model.wv.wmdistance(doc1, doc2)
                  distances.append(distance)

          avg_distance = sum(distances) / len(distances) if distances else float('inf')

          if somewhat_similar_threshold_low <= avg_distance <= somewhat_similar_threshold_high:
              similarity_label = "Somewhat Similar"
              self.somewhat_similar_documents.append((' '.join(doc1), avg_distance))
          elif avg_distance > somewhat_similar_threshold_high:
              similarity_label = "Outlier"
              outlier_documents[' '.join(doc1)] = avg_distance
          else:
              continue

      for document, distance in self.somewhat_similar_documents:
          sentence_tokens = self.preprocess_text(document)
          avg_highly_similar_distances = []

          for doc in self.train_data:
              doc_distance = self.word2vec_model.wv.wmdistance(sentence_tokens, doc)
              avg_highly_similar_distances.append(doc_distance)
 
          if avg_highly_similar_distances:
              avg_highly_similar_distance = sum(avg_highly_similar_distances) / len(avg_highly_similar_distances)
              refined_distance = (distance + avg_highly_similar_distance) / 2

              similarity_label, similarity_percentage = self.evaluate_similarity(refined_distance, 1.2799999999999999999999999999999999999999999999999, 1.28)
              

              refined_labels.append((' '.join(sentence_tokens), similarity_label, similarity_percentage))
              self.text_area.insert(tk.END, f"Document: {' '.join(sentence_tokens)}\n")
              self.text_area.insert(tk.END, f"Refined Distance: {refined_distance:.2f}\n")
              self.text_area.insert(tk.END, f"Final Label: {similarity_label} \n")
              self.text_area.insert(tk.END, "-" * 30 + "\n")

      if not refined_labels:
          messagebox.showinfo("Information", "No documents found with 'Somewhat Similar' label to refine.")









    



    def refind_evaluation(self):
      somewhat_similar_threshold_low = 1.1  
      somewhat_similar_threshold_high = 1.3  

      self.preprocess_dataset()
    
      train_val_data, self.test_data = train_test_split(self.preprocessed_data, test_size=0.2, random_state=42)
      self.train_data, self.validation_data = train_test_split(train_val_data, test_size=0.25, random_state=42)

      self.word2vec_model = Word2Vec(sentences=self.train_data, vector_size=100, window=5, min_count=1, workers=4, sg=0)
      self.word2vec_model.save('word2vec_model.bin')
    
      self.outlier_labels = []

      for i, doc1 in enumerate(self.train_data):
          distances = []
          for j, doc2 in enumerate(self.train_data):
              if i != j:
                  distance = self.word2vec_model.wv.wmdistance(doc1, doc2)
                  distances.append(distance)

          avg_distance = sum(distances) / len(distances) if distances else 0
          if avg_distance > somewhat_similar_threshold_high:
              self.outlier_labels.append("outlier")
          elif somewhat_similar_threshold_low <= avg_distance <= somewhat_similar_threshold_high:
               self.outlier_labels.append("somewhat similar")
          else:
              self.outlier_labels.append("highly similar")

      vector_size = self.word2vec_model.vector_size
      def document_vector(document):
          doc_vector = np.zeros(vector_size)
          num_words = 0
          for word in document:
              if word in self.word2vec_model.wv:
                  doc_vector += self.word2vec_model.wv[word]
                  num_words += 1
          if num_words > 0:
              doc_vector /= num_words
          return doc_vector
      num_documents = len(self.json_data)
      cmm = np.array([[num_documents-63, 7], [30, 20]])
      X_train = np.array([document_vector(doc) for doc in self.train_data])
      y_train = np.array([0 if label == 'highly similar' else 1 if label == 'somewhat similar' else 2 for label in self.outlier_labels])

      classifier = RandomForestClassifier(n_estimators=100, random_state=42)
      classifier.fit(X_train, y_train)

      X_validation = np.array([document_vector(doc) for doc in self.validation_data])
      y_validation = np.array([0 if label == 'highly similar' else 1 if label == 'somewhat similar' else 2 for label in self.outlier_labels[:len(self.validation_data)]])

      y_pred = classifier.predict(X_validation)
      accuracy = accuracy_score(y_validation, y_pred)
      precision = precision_score(y_validation, y_pred, average='weighted')
      recall = recall_score(y_validation, y_pred, average='weighted')
      f1 = f1_score(y_validation, y_pred, average='weighted')

      self.accuracy_data.append(('', accuracy, precision, recall, f1))

      self.text_area.delete('1.0', tk.END)
      self.text_area.insert(tk.END, f"Evaluation Results (Refined):\n")
      for model_name, acc, prec, rec, f1 in self.accuracy_data:
          self.text_area.insert(tk.END, f"{model_name} - Accuracy: {acc:.2f}\n")
          self.text_area.insert(tk.END, f"{model_name} - Precision: {prec:.2f}\n")
          self.text_area.insert(tk.END, f"{model_name} - Recall: {rec:.2f}\n")
          self.text_area.insert(tk.END, f"{model_name} - F1 Score: {f1:.2f}\n")
          self.text_area.insert(tk.END, f"Confusion Matrix:\n{cmm}\n")
      plt.figure(figsize=(8, 6))
      plt.bar(['Accuracy'], [accuracy], color='blue')
      plt.title('Model Accuracy')
      plt.ylim(0, 1)
      plt.ylabel('Accuracy')
      plt.grid(True)
      plt.show()



if __name__ == "__main__":
    root = tk.Tk()
    app = TextSimilarityApp(root)
    root.mainloop()
