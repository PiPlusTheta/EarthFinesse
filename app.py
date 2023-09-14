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

# Create a folder to store PDF reports
pdf_reports_folder = "pdf_reports"
os.makedirs(pdf_reports_folder, exist_ok=True)

# Create a folder to store temporary images
temp_images_folder = "temp_images"
os.makedirs(temp_images_folder, exist_ok=True)

# Load model
model = load_model('terrain__2023_09_13__11_52_06___Accuracy_0.9787.h5')

# Class labels
label_map = {0: 'Grassy', 1: 'Marshy', 2: 'Rocky', 3: 'Sandy'}

# Define military camo theme colors
bg_color = "#383838"  # Dark background
text_color = "#FFFFFF"  # White text
accent_color = "#4CAF50"  # Military green accent color

# Apply the theme
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

class PDF(FPDF):
    def header(self):
        # Add a border to the header of each page
        self.rect(5.0, 5.0, 200.0, 15.0)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, "EarthFinesse Military Terrain Classification Report", ln=True, align="C")
        self.cell(0, 10, "", ln=True)  # Add an empty line

    def footer(self):
        # Add a border to the footer of each page
        self.set_y(-15.0)
        self.rect(5.0, 287.0, 200.0, 15.0)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')

def generate_common_explanation(terrain):
    explanations = {
        'Grassy': """The terrain is classified as grassy due to its abundant vegetation and distinct characteristics. It features a rich carpet of greenery, consisting of various types of grasses, shrubs, and other plants. This lush vegetation indicates a favorable environment for plant growth, typically found in grassy plains, meadows, or dense forests. Grassy terrain is known for its capacity to provide excellent cover for both military operations and wildlife, making it a valuable asset in various scenarios.

In military operations, grassy terrain can be advantageous for several reasons. The dense vegetation offers natural concealment, allowing troops to use the environment as cover and remain hidden from adversaries. Additionally, it provides sustenance for wildlife and can be a reliable source of food for military personnel in certain situations. However, it may also pose challenges, such as limited visibility and obstacles for vehicle movement in densely vegetated areas. The model recognized these characteristics and classified the terrain accordingly.""",

        'Marshy': """The terrain is identified as marshy based on its unique attributes, which include the presence of standing water, soft and muddy ground, and aquatic vegetation. Marshy terrain is often characterized by its challenging and waterlogged conditions, making it difficult to traverse. It is commonly found in coastal regions, riverbanks, or low-lying areas where water accumulates.

In military operations, marshy terrain presents several distinctive challenges. Its soft, muddy ground can bog down vehicles and hinder infantry movement, reducing mobility and speed. The presence of water bodies creates natural obstacles that require careful navigation. However, marshy terrain can also offer some advantages for defense due to its difficult terrain and limited mobility for adversaries. The model recognized these wetland features, leading to the classification of marshy terrain. This type of terrain may pose significant challenges for military operations, as it can limit mobility and visibility while providing natural obstacles for defense.""",

        'Rocky': """The terrain classification as rocky is based on its rugged and uneven surface, which is characterized by the abundance of large rocks, boulders, and irregular features. Rocky terrain is often challenging to navigate and can significantly limit vehicle movement. It is typically found in mountainous regions, rocky outcrops, or areas with substantial geological formations.

In a military context, rocky terrain presents a mixed set of advantages and disadvantages. The ruggedness of the terrain offers natural defensive advantages, as troops can find cover behind rocks and boulders. Additionally, the uneven surface can make it challenging for adversaries to advance. However, rocky terrain can also impede troop movement and require specialized tactics and equipment for military operations. The model identified these distinct features, leading to the classification of rocky terrain.""",

        'Sandy': """The terrain classification as sandy is attributed to its predominant sandy composition, characterized by loose, granular soil and a lack of significant vegetation. Sandy terrain is commonly found in desert regions, coastal dunes, or arid environments.

In military operations, sandy terrain presents unique challenges. The loose sand can impede vehicle movement and create conspicuous tracks, making it easier for adversaries to detect and track the movement of military units. Additionally, the lack of vegetation limits natural concealment options. However, sandy terrain can also offer some natural cover in the form of sand dunes and the ability to dig defensive positions in the soft soil. Military operations in sandy terrain often require specific equipment and strategies to address the unique challenges it poses. The model recognized these sandy characteristics, leading to the classification of sandy terrain.""",

        # Add more terrain explanations here...
    }
    
    return explanations.get(terrain, "The terrain is classified as an unknown type.")

def classify_image(img):
    img = img.resize((224, 224))
    
    # Ensure the image has 3 color channels (RGB)
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

        # Save the image to the temporary images folder in PNG format
        img_path = os.path.join(temp_images_folder, f"temp_image_{i}.png")
        img = Image.open(row['Image'])
        img.save(img_path, format="PNG")

        pdf.ln(10)
        pdf.image(img_path, x=20, w=80)
        pdf.ln(10)
        pdf.cell(0, 10, f"Predicted Terrain Type: {terrain}", ln=True)
        pdf.cell(0, 10, f"Prediction Confidence: {confidence * 100:.2f}%", ln=True)
        
        # Generate a common explanation for the terrain
        explanation = generate_common_explanation(terrain)
        pdf.multi_cell(0, 10, f"Terrain Explanation: {explanation}")

    pdf_file_path = os.path.join(pdf_reports_folder, f"terrain_classification_report_{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf")
    pdf.output(pdf_file_path)

    return pdf_file_path

def main():
    st.sidebar.title('Select Operation')
    operation = st.sidebar.radio("Choose an operation:", ("File Upload",))

    if operation == "File Upload":
        st.title('🌍 EarthFinesse - Military Terrain Classifier 🛡️')
        st.header('File Upload and Classification')
        
        st.sidebar.header('Mission Settings')
        threshold = st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.5, 0.01)
        show_probabilities = st.sidebar.checkbox('Show Probabilities', False)
        bulk_classification = st.sidebar.checkbox('Bulk Classification', False)

        if bulk_classification:
            st.sidebar.write("Upload multiple images to classify in bulk.")
        else:
            st.sidebar.write("Upload a single image for classification.")

        uploaded_files = st.file_uploader('Upload Reconnaissance Images', type=['png', 'jpg'], accept_multiple_files=True, help="Select one or more reconnaissance images.")

        if uploaded_files:
            bulk_results = []
            st.markdown("---")

            for i, file in enumerate(uploaded_files):
                st.subheader(f"Image {i+1}")
                st.image(file, caption="Reconnaissance Image", use_column_width=True)

                terrain, confidence = classify_image(Image.open(file))

                # Display the prediction and confidence score to the right of the image
                st.write("### Prediction:")
                st.write(f"🌲 Predicted Terrain Type: {terrain}")
                st.write(f"🎯 Prediction Confidence: {confidence * 100:.2f}%")

                # Generate a common explanation for the terrain
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
