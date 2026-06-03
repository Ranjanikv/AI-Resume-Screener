# AI-Resume-Screener
AI-powered Resume Screener and Ranker built with Streamlit. Upload a Job Description and multiple resumes to automatically rank candidates based on skill matching and TF-IDF cosine similarity.
# AI Resume Screener & Ranker

## Overview

AI Resume Screener is a Streamlit-based web application that helps recruiters and hiring teams quickly evaluate resumes against a job description. The application extracts text from resumes and job descriptions, identifies relevant skills, and ranks candidates based on similarity scores.

## Features

* Upload Job Descriptions (PDF, DOCX, TXT)
* Upload Multiple Resumes
* Automatic Skill Detection
* Hard Skills and Soft Skills Matching
* Resume Ranking using TF-IDF and Cosine Similarity
* Simple and Interactive Streamlit Interface

## Tech Stack

* Python
* Streamlit
* Scikit-learn
* PDFPlumber
* Docx2txt
* NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/AI-Resume-Screener.git
cd AI-Resume-Screener
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```

## How It Works

1. Upload a Job Description.
2. Upload one or more resumes.
3. Click **Run AI Screening**.
4. The system:

   * Extracts text from documents.
   * Detects required skills from the job description.
   * Matches skills in resumes.
   * Calculates similarity scores using TF-IDF and Cosine Similarity.
   * Ranks candidates from best to least match.

## Future Improvements

* Resume feedback and improvement suggestions
* Experience and education scoring
* AI-generated candidate summaries
* Support for more document formats

## License

MIT License
