# Book Recommendation Engine

A machine learning project that recommends books similar to a given title using **k-NN (k-nearest neighbours)** with **cosine similarity**.

The engine is trained on the [Book-Crossings dataset](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip), which contains thousands of user ratings on books, and suggests titles based on books that have been liked by similar users.

---

# Description

The project uses a collaborative filtering recommender built from scratch using Python.
It:
- Processes and filters the **Book-Crossings dataset** to keep active users and popular books.
- Builds a **book-user matrix** where each row is a book, and each column is a user.
- Calculates the **cosine similarity** between books to find those with similar preferences.
- Uses **k-NN** to recommend books similar to a given title.
- Includes evaluation using **Recall@K** to measure how often the model correctly recommends books a user may like.

---

# How it Works
1. **Loading the Data**: The Book-Crossings dataset is downloaded and loaded.
2. **Filtering:** Removes all users with less than `min_user_ratings` ratings, and all books with less than `min_book_ratings` ratings.
3. **Building the Matrix**: Creates a binary book x user matrix, where 1 represents a user rating a book positively, and 0 represents otherwise. It may also apply TF-IDF like weighting to try to reduce popularity bias.
4. **Training:** Fits a k-NN model using cosine similarity on the matrix.
5. **Evaluation:** Splits data by holding out 1 positive rating per user, and measuring how often the held out book appears in the top K neighbours (Recall@K).
6. **Recommendation**: Given a title, finds the most similar books based on the patterns of the other readers.

---

# Dataset
As mentioned, we use the **Book-Crossings** dataset. It contains:

- ~270,000 books.
- ~78,000 users.
- ~1,000,000 ratings.

Within, it contains three CSV files:
- `BX-Books.csv`, containing ISBN, title, and author.
- `BX-Book-Ratings.csv`, containing user ID, ISBN, and rating.
- `BX-Users.csv`, containing user details. This file is not used.

---

# Usage

### 1. Clone the repo.
```bash
git clone https://github.com/cadhann/book_recommendation_engine.git
cd book_recommendation_engine
```

### 2. Set up environment.
```bash
python -m venv .venv
# On Powershell
.venv\Scripts\Activate.ps1
# In Command Prompt
.venv\Activate.bat
# In Bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Train the recommender.
```bash
python -m scripts.train --min-user-ratings 200 --min-book-ratings 100 --k 5
```
This will:
- Download & preprocess the data.
- Build the item x user matrix.
- Train the k-NN model.
- Evaluate it using Recall@K.
- Save the model & matrix to artifacts/

Output example:
```
Downloading data (if required)...
Loading CSVs...
Filtering active users and popular books...
Building item-user matrix (titles x users)...
Training k-NN model...
Evaluating...
Recall@5: 0.462
Saving artifacts...
Done.
```

### 4. Generate recommendations.
When trained, similar books can be queried.
```bash
python -m scripts.recommend --title "Animal Farm" --top-k 5
```

Output example:
```
Loading artifacts...
Base title: Animal Farm
1) 1984 (distance: 0.70)
2) The Corrections: A Novel (distance: 0.75)
3) To Kill a Mockingbird (distance: 0.83)
4) Lord of the Flies (distance: 0.83)
5) Shopgirl : A Novella (distance: 0.83)
```

The lower the distance, the more similar the book. Here, we can see that 1984 (by the same author as Animal Farm, George Orwell) is matched, along with To Kill a Mockingbird and Lord of the Flies, two novels with similar themes to Animal Farm.