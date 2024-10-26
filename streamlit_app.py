#https://towardsdatascience.com/how-to-extract-keywords-from-pdfs-and-arrange-in-order-of-their-weights-using-python-841556083341
import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from groq import Groq
import json 
import pandas as pd


groq_llama_api_key=st.secrets.api_key
client = Groq(
    api_key=groq_llama_api_key,
)

skills_list = [
    # Programming Languages
    "Python", "Java", "JavaScript", "C#", "C++", "R", "Go", "Ruby", "PHP", "Swift", 
    "Kotlin", "TypeScript", "HTML", "CSS", "Scala", "SQL",

    # Data Science & Machine Learning
    "Data Analysis", "Data Science", "Machine Learning", "Deep Learning", "AI", 
    "Natural Language Processing", "Computer Vision", "Big Data", "Data Visualization",
    "Neural Networks", "TensorFlow", "Keras", "PyTorch", "scikit-learn", 
    "Pandas", "NumPy", "Matplotlib", "Seaborn", "Tableau", "Power BI",

    # Cloud & DevOps
    "AWS", "Azure", "Google Cloud", "Cloud Computing", "Docker", "Kubernetes", 
    "CI/CD", "Jenkins", "Terraform", "Ansible", "Git", "GitHub", "GitLab", "Linux", "Bash", "DevOps",

    # Web Development & Frameworks
    "Django", "Flask", "FastAPI", "React", "Vue.js", "Angular", "Node.js", 
    "Express", "Next.js", "Svelte", "Spring Boot", "Laravel", "ASP.NET", 
    "GraphQL", "REST APIs", "Full Stack Development",

    # Databases
    "PostgreSQL", "MySQL", "MongoDB", "Cassandra", "Redis", "Elasticsearch", 
    "Oracle", "SQL Server", "SQLite", "ETL", "Hadoop", "Spark",

    # Cybersecurity & Networking
    "Cybersecurity", "Penetration Testing", "Network Security", "Cryptography", 
    "Firewalls", "VPN", "Ethical Hacking", "Cloud Security", "ISO 27001",

    # Project Management & Agile Methodologies
    "Project Management", "Agile", "Scrum", "Kanban", "JIRA", "Confluence", 
    "Trello", "Asana", "Time Management", "Risk Management",

    # Soft Skills
    "Communication", "Teamwork", "Leadership", "Problem-Solving", "Critical Thinking", 
    "Adaptability", "Creativity", "Collaboration", "Empathy", "Presentation Skills",

    # Industry-Specific Tools & Skills
    "Financial Analysis", "Salesforce", "CRM", "SEO", "Digital Marketing", 
    "Content Marketing", "Email Marketing", "Product Management", "UX/UI", 
    "Figma", "Wireframing", "Prototyping", "Usability Testing",

    # Additional Technical Skills
    "Automation", "IoT", "Blockchain", "VR", "AR", "Unity", 
    "Mobile Development", "React Native", "Flutter", "API Development", 
    "Microservices", "Software Engineering", "Quality Assurance", "Test Automation", "Selenium"
]

def extract_skills(keywords, skills_list):
    keywords_lower = [keyword.lower() for keyword in keywords]
    skills_lower = [skill.lower() for skill in skills_list]
    identified_skills = [skill.capitalize() for skill in skills_lower if skill in keywords_lower]
    return identified_skills

def render_json(json_data, level=0):
    json_data = json.loads(json_data)
    for qa in json_data :
        with st.expander(qa['question']):
            st.write(qa['answer'])



def extract_text_from_pdf(pdf):
    pdf_reader = PyPDF2.PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_keywords(text, num_keywords=10):
    # Using TF-IDF to extract keywords
    vectorizer = TfidfVectorizer(max_features=num_keywords, stop_words='english')
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords

def generate_response(input_text):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": input_text ,
            }
        ],
        model="llama3-8b-8192" )

    response_content = chat_completion.choices[0].message.content
    return response_content


st.title("PDF Keyword Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    text = extract_text_from_pdf(uploaded_file)

    keywords = extract_keywords(text)
    
    skills = extract_skills(keywords, skills_list)

    skill_set = ", ".join(skills)
    
    prompt = 'Give me 15 questions and answers in a pattern of key value pair in JSON FORMAT the object should have `question` and  `answer` as its index  ,  based on these skillsets :' + skill_set + ' only return me the array no extra text  than that.'

    json_response = generate_response(prompt)

    # Display the JSON data in a collapsible format if it is valid
    if json_response:
        st.subheader("Generated Questions and Answers")
        render_json(json_response)

    




