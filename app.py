import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os
from datetime import datetime
from fpdf import FPDF
import pandas as pd
import base64

pdf_reports_folder = "pdf_reports"
os.makedirs(pdf_reports_folder, exist_ok=True)

temp_images_folder = "temp_images"
os.makedirs(temp_images_folder, exist_ok=True)

model = load_model('terrain__2023_09_13__11_52_06___Accuracy_0.9787.h5')

label_map = {0: 'Grassy', 1: 'Marshy', 2: 'Rocky', 3: 'Sandy'}

bg_color = "#383838"  
text_color = "#FFFFFF"  
accent_color = "#4CAF50"  

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background-color: {bg_color};
        color: {text_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {bg_color};
    }}
    .widget-label {{
        color: {text_color};
    }}
    .stButton>button {{
        color: {text_color};
        background-color: {accent_color};
    }}
    </style>
    """,
    unsafe_allow_html=True
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

class PDF(FPDF):
    def header(self):
        self.rect(5.0, 5.0, 200.0, 15.0)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, "EarthFinesse Military Terrain Classification Report", ln=True, align="C")
        self.cell(0, 10, "", ln=True) 

    def footer(self):
        self.set_y(-17.0)
        self.rect(5.0, 278.0, 200.0, 15.0)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, align='C')

def generate_common_explanation(terrain):
    explanations = {
        'Grassy': """The terrain is classified as grassy based on several visual cues. It displays a lush, green carpet of vegetation, including various types of grasses and other plants. The presence of dense vegetation indicates a favorable environment for plant growth, typically found in grassy plains, meadows, or forests. This terrain type offers good cover for camouflage and can provide sustenance for wildlife and troops in certain situations. The model recognized these characteristics and classified the terrain accordingly.""",

        'Marshy': """The terrain is identified as marshy due to its distinctive characteristics. It exhibits signs of wetland features, such as standing water, soft, muddy ground, and aquatic vegetation. Marshy terrain is often challenging to traverse and may impede movement. It can be found in coastal areas, riverbanks, or low-lying regions. The model recognized the presence of these wetland features, leading to the classification of marshy terrain. This type of terrain may pose challenges for military operations, as it can limit mobility and visibility while providing natural obstacles for defense.""",

        'Rocky': """The terrain classification as rocky is based on its rugged and uneven surface. It is characterized by the presence of large rocks, boulders, and irregular terrain. Rocky terrain is often challenging to navigate and may limit vehicle movement. It can be found in mountainous regions, rocky outcrops, or areas with substantial geological formations. The model identified these distinct features, leading to the classification of rocky terrain. While rocky terrain can offer natural defensive advantages, it can also impede troop movement and require specialized tactics for military operations.""",

        'Sandy': """The terrain classification as sandy is attributed to its predominant sandy composition. Sandy terrain lacks significant vegetation and is characterized by loose, granular soil. It is commonly found in desert regions, coastal dunes, or arid environments. Sandy terrain can present challenges for both mobility and concealment, as the loose sand can impede vehicle movement and leave conspicuous tracks. The model recognized these sandy characteristics, leading to the classification of sandy terrain. Military operations in sandy terrain often require specific equipment and strategies to address the unique challenges it poses.""",

    }
    
    return explanations.get(terrain, "The terrain is classified as an unknown type.")

def classify_image(img):
    img = img.resize((224, 224))
    
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    prediction = model.predict(x)
    label_index = np.argmax(prediction)
    prediction_prob = prediction[0, label_index]
    return label_map[label_index], prediction_prob

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

def generate_pdf_report(df):
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="EarthFinesse Military Terrain Classification Report", ln=True, align="C")

    for i, row in df.iterrows():
        terrain = row['Terrain']
        confidence = row['Confidence']
        explanation = generate_common_explanation(terrain)

        img_path = os.path.join(temp_images_folder, f"temp_image_{i}.png")
        img = Image.open(row['Image'])
        img.save(img_path, format="PNG")

        pdf.ln(10)
        pdf.image(img_path, x=20, w=80)
        pdf.ln(10)
        pdf.cell(0, 10, f"Predicted Terrain Type: {terrain}", ln=True)
        pdf.cell(0, 10, f"Prediction Confidence: {confidence * 100:.2f}%", ln=True)
        
        explanation = generate_common_explanation(terrain)
        pdf.multi_cell(0, 10, f"Terrain Explanation: {explanation}")

    pdf_file_path = os.path.join(pdf_reports_folder, f"terrain_classification_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    pdf.output(pdf_file_path)

    return pdf_file_path

def main():
    st.sidebar.title('Select Operation')
    st.title('🌍 EarthFinesse - Military Terrain Classifier 🛡️')
    st.header('Bulk Image Classification')
    
    st.sidebar.header('Mission Settings')
    threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)
    show_probabilities = st.sidebar.checkbox('Show Probabilities', False)

    uploaded_files = st.file_uploader('Upload Reconnaissance Images', type=['png', 'jpg'], accept_multiple_files=True, help="Select one or more reconnaissance images.")

    if uploaded_files:
        bulk_results = []
        st.markdown("---")

        for i, file in enumerate(uploaded_files):
            st.subheader(f"Image {i+1}")
            st.image(file, caption="Reconnaissance Image", use_column_width=True)

            terrain, confidence = classify_image(Image.open(file))

            st.write("### Prediction:")
            st.write(f"🌲 Predicted Terrain Type: {terrain}")
            if show_probabilities:
                st.write(f"🎯 Prediction Confidence: {confidence * 100:.2f}%")

            explanation = generate_common_explanation(terrain)
            st.write("### Terrain Explanation:")
            st.write(explanation)

            bulk_results.append({
                'Image': file,
                'Terrain': terrain,
                'Confidence': confidence
            })

        if bulk_results:
            st.sidebar.markdown("---")
            st.sidebar.header('Generate PDF Report')
            if st.sidebar.button("Download PDF Report"):
                df = pd.DataFrame(bulk_results)
                pdf_file_path = generate_pdf_report(df)
                st.sidebar.markdown(get_binary_file_downloader_html(pdf_file_path, 'Download PDF'), unsafe_allow_html=True)
                st.sidebar.success("PDF report generated!")

if __name__ == "__main__":
    main()
