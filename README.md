# REBA Assessment System

Ergonomic risk assessment system using Fine-tuned YOLOv11m pose estimation 
model combined with MiDaS depth estimation model to generate 3D coordinates 
and calculate REBA scores for construction workers (Bricklaying, Plastering, 
Tile Installation).

## Setup
1. Clone this repo
   git clone https://github.com/Yoon1414/REBA-Assessment-System.git

2. Install dependencies
   pip install -r requirements.txt

3. Download model weights and place best.pt in project root
   👉 [Download best.pt](YOUR_GOOGLE_DRIVE_LINK_HERE)

4. Run the app
   streamlit run app.py
