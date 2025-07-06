# Semantic Analysis with Embeddings and LDA

## Overview
This repository contains the implementation of the Semantic Analysis and Topic Modeling project, an academic project completed for the CPSC 330: Applied Machine Learning course. The project explores natural language processing (NLP) techniques, specifically word embeddings and topic modeling, using the GloVe Wikipedia pre-trained embeddings and the 20 Newsgroups dataset. The project is implemented in a Jupyter Notebook (`semantic_analysis.ipynb`) and focuses on analyzing word similarities, biases in embeddings, and discovering topics in text data.

## Objectives
The project is divided into three main exercises:
1. **Exploring Pre-trained Word Embeddings**: Using GloVe Wikipedia embeddings (`glove-wiki-gigaword-100`) to analyze word similarities, check vocabulary coverage, and investigate potential biases.
2. **Topic Modeling**: Applying Latent Dirichlet Allocation (LDA) on the 20 Newsgroups dataset to discover high-level themes and evaluate their alignment with known categories.
3. **Short Answer Questions**: Addressing concepts related to recommender systems and transfer learning in NLP.

## Dataset
- **GloVe Wikipedia Embeddings**: Pre-trained word vectors (`glove-wiki-gigaword-100`) with 400,000 word representations, used for word similarity and bias analysis.
- **20 Newsgroups Dataset**: A collection of approximately 20,000 newsgroup documents, subset to 8 categories (e.g., `rec.sport.hockey`, `talk.politics.guns`, `comp.graphics`) with 4,563 training documents. The dataset is preprocessed to remove headers, footers, and quotes.

## Project Structure
The notebook (`semantic_analysis.ipynb`) is organized into three exercises:

### Exercise 1: Exploring Pre-trained Word Embeddings
- **1.1 Word Similarity**: Identified similar words for a chosen list (`burger`, `boxing`, `statistics`, `iphone`) using GloVe embeddings.
- **1.2 Cosine Similarity**: Calculated cosine similarity for word pairs (e.g., `coast` vs. `shore`, `dog` vs. `cat`) to assess semantic relationships.
- **1.3 Vocabulary Coverage**: Checked if neologisms and biomedical abbreviations (e.g., `covididiot`, `pxg`) are represented in the GloVe vocabulary.
- **1.4 Stereotypes and Biases**: Explored biases in embeddings using analogies (e.g., `man:doctor :: woman:?`, `urban:educated :: rural:?`) to identify potential stereotypes.
- **1.5 Discussion**: Analyzed observed biases (e.g., associating `mexican` with `traffickers`, `rural` with `illiterate`) and discussed their real-world implications, such as perpetuating negative stereotypes.

### Exercise 2: Topic Modeling
- **2.1 Preprocessing**: Preprocessed the `text` column of the 20 Newsgroups dataset using spaCy, creating a `text_pp` column by removing stop words, short tokens, irrelevant POS tags, and applying lemmatization.
- **2.2 Justification**: Justified preprocessing steps to enhance topic modeling by focusing on semantically meaningful tokens.
- **2.3 LDA Model**: Built LDA models with `n_components=8` (aligned with the 8 categories) using `scikit-learn`'s `LatentDirichletAllocation` to discover topics.
- **2.4 Word-Topic Association**: Identified top 10 words per topic and assigned labels (e.g., "Hockey Teams and Leagues", "Global Politics").
- **2.5 Document-Topic Association**: Analyzed topic assignments for the first five documents, noting mostly accurate alignments with minor errors (e.g., a `comp.windows.x` document assigned to "Law and Rights").

### Exercise 3: Short Answer Questions
- Explained content-based filtering in recommender systems, its negative consequences (e.g., information bubbles, bias propagation), and transfer learning in NLP.

## Key Findings
- **Word Embeddings**: GloVe embeddings effectively captured semantic similarities (e.g., `dog` and `cat` with high similarity) but showed biases, such as gender stereotypes (`man:doctor :: woman:nurse`) and problematic associations (`mexican:traffickers`).
- **Topic Modeling**: The LDA model with 8 topics aligned well with dataset categories, though some documents were misassigned due to vocabulary overlap or preprocessing limitations.
- **Preprocessing Impact**: Removing irrelevant POS tags and stop words improved topic coherence, but some noise remained, suggesting potential for further preprocessing refinement.

## Potential Improvements
- **Preprocessing**: Enhance preprocessing by handling URLs, numbers, and domain-specific terms to improve topic clarity.
- **LDA Tuning**: Experiment with different `n_components` values or increase iterations for better topic separation.
- **Bias Mitigation**: Apply debiasing techniques to embeddings to reduce harmful stereotypes.
- **Dataset Size**: Use the full 20 Newsgroups dataset or additional categories for more robust topic modeling.

## Dependencies
The notebook requires the following Python libraries:
- `otter`
- `matplotlib`
- `numpy`
- `pandas`
- `scikit-learn`
- `gensim`
- `spacy` (with `en_core_web_md` model)
- `mglearn`

## Setup Instructions
To run the notebook on your own machine, follow these steps:

1. **Install Python**: Ensure Python 3.8 or higher is installed. Download from [python.org](https://www.python.org/downloads/) if needed.
2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:
   Install the required libraries using pip:
   ```bash
   pip install pandas matplotlib numpy scikit-learn gensim spacy mglearn otter-grader
   ```
   Install the spaCy language model:
   ```bash
   python -m spacy download en_core_web_md
   ```
4. **Verify Installation**: Ensure all libraries are installed by running `pip list` and checking for the listed dependencies.

## How to Run
1. **Download Dataset**: The 20 Newsgroups dataset is automatically fetched using `sklearn.datasets.fetch_20newsgroups`. No manual download is required.
2. **Download the Notebook**: Clone this repository or download `semantic_analysis.ipynb` to your machine.
3. **Run Jupyter Notebook**:
   - Start Jupyter Notebook:
     ```bash
     jupyter notebook
     ```
   - Open `semantic_analysis.ipynb` in the Jupyter interface.
4. **Execute the Notebook**:
   - Run all cells (`Kernel -> Restart Kernel and Clear All Outputs`, then `Run -> Run All Cells`) to execute the code and generate outputs.
5. **View Results**: Outputs include word similarity results, cosine similarities, vocabulary checks, bias analyses, topic word lists, document topic assignments, and short answer responses.
6. **Optional**: Save the preprocessed DataFrame to a CSV (e.g., `df.to_csv('preprocessed_data.csv')`) to avoid re-running spaCy preprocessing.

## Notes
- **Environment**: Tested with Python 3.12.0, but compatible with Python 3.8 or higher.
- **Preprocessing Time**: spaCy preprocessing may be slow for the full dataset. Consider saving preprocessed data to a CSV for efficiency.
- **Output Rendering**: Ensure plots and outputs are rendered correctly. Export to PDF or HTML if the notebook is too large for submission platforms like Gradescope.
- **Submission**: If submitting for a course, follow the specific submission guidelines (e.g., Gradescope instructions) provided by your instructor.

## Acknowledgments
- GloVe embeddings provided by Stanford University.
- 20 Newsgroups dataset provided by scikit-learn.
- Code for preprocessing adapted from CPSC 330 lecture notes.

