import streamlit as st
import docx2txt
import pdfplumber
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re  


MASTER_TECH_LIST = [
    # --- UI/UX & Web Design ---
    'Figma', 'Adobe XD', 'Sketch', 'InVision', 'Zeplin', 'Miro', 
    'Wireframing', 'Prototyping', 'User Flows', 'Information Architecture',
    'Typography', 'Design Systems', 'Balsamiq', 'Framer',
    
    # --- Front-End Development ---
    'HTML', 'CSS', 'JavaScript', 'React', 'Angular', 'Vue', 'Next.js',
    'Tailwind CSS', 'Bootstrap', 'jQuery', 'TypeScript', 'SASS',
    
    # --- Back-End Development ---
    'Node.js', 'Python', 'Java', 'C++', 'Spring Boot', 'Django', 
    'Flask', 'Express.js', 'PHP', 'Laravel', 'Ruby on Rails', 
    'REST APIs', 'GraphQL', 'Microservices',
    
    # --- Mobile App Development (Flutter & Native) ---
    'Flutter', 'Dart', 'React Native', 'Swift', 'Kotlin', 'Objective-C',
    'Android Studio', 'Xcode', 'Mobile UI', 'Firebase',
    
    # --- Data Science & AI ---
    'Python', 'R', 'Machine Learning', 'TensorFlow', 'PyTorch', 
    'Scikit-learn', 'NLP', 'Prompt Engineering', 'LLMs', 'Deep Learning',
    'Computer Vision', 'Hugging Face', 'LangChain',
    
    # --- Data Analysis & BI ---
    'SQL', 'Excel', 'Power BI', 'Tableau', 'Pandas', 'NumPy',
    'Data Visualization', 'Google Analytics', 'Data Cleaning', 'Reporting',
    
    # --- Finance & Accounting ---
    'Bookkeeping', 'Financial Modeling', 'Auditing', 'Tax Preparation', 'Payroll',
    'Budgeting', 'QuickBooks', 'Balance Sheets', 'P&L Management', 'Forecasting',
    
    # --- Marketing & Content ---
    'SEO', 'Copywriting', 'Google Analytics', 'Content Strategy', 'Social Media Management',
    'Email Marketing', 'Market Research', 'CRM', 'HubSpot', 'Salesforce', 'Branding',
    
    # --- Sales & Business Development ---
    'Cold Calling', 'Lead Generation', 'B2B Sales', 'Negotiation', 'Account Management',
    'Sales Pitching', 'Deal Closing', 'Pipeline Management', 'Contract Negotiation',
    
    # --- HR & Operations ---
    'Recruiting', 'Talent Acquisition', 'Onboarding', 'Labor Laws', 'Conflict Resolution',
    'Scheduling', 'Logistics', 'Supply Chain', 'Inventory Management', 'Process Optimization'
]

MASTER_SOFT_LIST = [
    # --- Modern Workplace & AI Era Skills ---
    'Digital Fluency', 'Adaptability', 'Continuous Learning', 'Creative Thinking',
    'Critical Thinking', 'Complex Problem-Solving', 'Analytical Thinking',
    'Ethical Judgment', 'Digital Mindset', 'Learning Agility',
    
    # --- Emotional & Interpersonal Intelligence ---
    'Emotional Intelligence', 'Empathy', 'Active Listening', 'Conflict Resolution', 
    'Psychological Safety', 'Stress Management', 'Cultural Intelligence',
    
    # --- Execution & Leadership ---
    'Decision Making', 'Resource Allocation', 'Team Building', 'Mentorship',
    'Transformational Leadership', 'Strategic Thinking', 'Remote Team Leadership',
    'Time Management', 'Accountability',
    
    # --- Collaboration & Organization ---
    'Communication', 'Collaborated', 'Design Thinking', 'Agile',  
    'Workshops', 'Teamwork', 'Cross-Functional Teamwork', 'Stakeholder Management'
]


class DocumentParser:
    """Handles text extraction for uploaded file streams."""
    
    @staticmethod
    def extract_text_from_pdf(uploaded_file):
        text = ""
        try:
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
        return text

    @staticmethod
    def extract_text_from_docx(uploaded_file):
        try:
            return docx2txt.process(uploaded_file)
        except Exception as e:
            st.error(f"Error reading DOCX: {e}")
            return ""

    def get_text(self, uploaded_file):
        if uploaded_file.name.endswith('.pdf'):
            return self.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.name.endswith('.docx'):
            return self.extract_text_from_docx(uploaded_file)
        elif uploaded_file.name.endswith('.txt'):
            return str(uploaded_file.read(), "utf-8")
        return ""


class ResumeScreener:
    """Calculates match scores and isolates skills found in the Job Description."""
    
    def __init__(self, tech_master_list, soft_master_list):
        self.tech_master = [skill.lower() for skill in tech_master_list]
        self.soft_master = [skill.lower() for skill in soft_master_list]

    def _skill_in_text(self, skill, text):
        """Helper to safely check for single letter skills or short words."""
        # For short or single-letter skills like R, C, or Go, use word boundaries
        if len(skill) <= 2:
            pattern = rf"\b{re.escape(skill)}\b"
            return re.search(pattern, text) is not None
        # For standard longer words, the simple 'in' check works fine
        return skill in text

    def extract_skills_from_jd(self, jd_text):
        jd_lower = jd_text.lower()
        active_tech = [skill for skill in self.tech_master if self._skill_in_text(skill, jd_lower)]
        active_soft = [skill for skill in self.soft_master if self._skill_in_text(skill, jd_lower)]
        return active_tech, active_soft

    def screen_resumes(self, jd_text, resume_texts, resume_names):
        jd_tech, jd_soft = self.extract_skills_from_jd(jd_text)
        
        vectorizer = TfidfVectorizer(stop_words='english')
        corpus = [jd_text] + resume_texts
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])[0]
        
        results = []
        for i, score in enumerate(scores):
            resume_lower = resume_texts[i].lower()
            
            # Use the boundary-aware helper for matching resumes
            matched_tech = [skill for skill in jd_tech if self._skill_in_text(skill, resume_lower)]
            matched_soft = [skill for skill in jd_soft if self._skill_in_text(skill, resume_lower)]
            
            results.append({
                'name': resume_names[i],
                'score': score,
                'tech_skills': list(set(matched_tech)),
                'soft_skills': list(set(matched_soft))
            })
            
        return results, jd_tech, jd_soft


def main():    
    st.set_page_config(page_title="AI Resume Screener", layout="centered")    
    st.title("AI Resume Screener & Ranker")
    st.write("Upload a Job Description and Candidate Resumes to automatically rank the best fits!")
    st.markdown("---")
    
    parser = DocumentParser()
    screener = ResumeScreener(MASTER_TECH_LIST, MASTER_SOFT_LIST)
    
    # 1. Job Description Upload
    st.subheader("1. Job Description")
    jd_file = st.file_uploader("Upload Job Description (PDF, DOCX, or TXT)", type=['pdf', 'docx', 'txt'], key="jd")
    
    # 2. Resumes Upload
    st.subheader("2. Candidate Resumes")
    resume_files = st.file_uploader("Upload Resumes (Multiple allowed)", type=['pdf', 'docx', 'txt'], accept_multiple_files=True, key="resumes")
    
    st.markdown("---")
    
    # 3. Action Button
    if st.button("Run AI Screening", type="primary"):
        if not jd_file:
            st.warning("Please upload a Job Description first!")
            return
        if not resume_files:
            st.warning("Please upload at least one resume!")
            return
            
        with st.spinner("Processing files and calculating scores..."):
            # Extract JD text
            jd_text = parser.get_text(jd_file)
            if not jd_text.strip():
                st.error("Job description could not be read!")
                return
                
            # Extract Resumes text
            resume_texts = []
            resume_names = []
            for file in resume_files:
                text = parser.get_text(file)
                if text.strip():
                    resume_texts.append(text)
                    resume_names.append(file.name)
                    
            if not resume_texts:
                st.error("No valid resumes could be read!")
                return
                
            # Run AI algorithm
            analyses, jd_tech, jd_soft = screener.screen_resumes(jd_text, resume_texts, resume_names)
            ranked_results = sorted(analyses, key=lambda x: x['score'], reverse=True)
            
            # Display Skills found in JD
            st.success("Analysis Complete!")
            
            st.markdown("### Skills Required by Job Description")
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Hard Skills Detected", value=len(jd_tech))
                st.write(", ".join(jd_tech) if jd_tech else "None detected")
            with col2:
                st.metric(label="Soft Skills Detected", value=len(jd_soft))
                st.write(", ".join(jd_soft) if jd_soft else "None detected")
                
            st.markdown("---")
            
            # Display Ranked Results
            st.markdown("### AI Ranked Resumes")
            for rank, res in enumerate(ranked_results, 1):
                match_percentage = round(res['score'] * 100, 2)
    
                with st.expander(f"{rank} | {res['name']} — {match_percentage}% Match"):
                    st.progress(res['score']) 
                    
                    st.markdown("**Matched Hard/Tech Skills:**")
                    st.write(", ".join(res['tech_skills']) if res['tech_skills'] else "None detected")
                    
                    st.markdown("**Matched Soft Skills:**")
                    st.write(", ".join(res['soft_skills']) if res['soft_skills'] else "None detected")
if __name__ == "__main__":
    main()