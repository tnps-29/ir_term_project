"""
Automatic Software Defect Classification System
Based on IEEE 1044-2009 Standard using Rocchio Algorithm (Nearest Centroid)

To run this application, ensure you have the required packages installed:
pip install streamlit pandas numpy pythainlp scikit-learn nltk

Then execute the following command in your terminal:
streamlit run app.py
"""
import os
import streamlit as st
import pandas as pd

from core.nlp import preprocess_text
from core.pipeline import train_models, evaluate_models

# ==========================================
# Application Configuration
# ==========================================
st.set_page_config(
    page_title="Software Defect Classifier", 
    page_icon="🐞",
    layout="wide"
)

def load_css():
    """Load external CSS from assets folder."""
    css_path = os.path.join(os.path.dirname(__file__), 'assets', 'style.css')
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ==========================================
# Streamlit Frontend Interface
# ==========================================
def main():
    # --- Custom CSS Styling ---
    load_css()

    # --- Header Section ---
    st.title("Automatic Software Defect Classification")
    st.markdown("""
    This system implements the **Rocchio Algorithm (Nearest Centroid)** to automatically 
    classify software defect reports into 4 dimensions according to the **IEEE 1044-2009** standard.
    """)
    
    # --- Sidebar Section ---
    with st.sidebar:
        st.header("About the System")
        st.markdown("""
        **IEEE 1044-2009 Dimensions:**
        - **Defect Type:** E.g., Logic, Interface, Data, Documentation
        - **Defect Effect:** E.g., System crash, Wrong calculation, UI glitch
        - **Severity:** E.g., Critical, Major, Minor
        - **Defect Mode:** E.g., Wrong, Missing, Extra
        
        **Machine Learning Pipeline:**
        1. **Data Cleaning:** Blacklist filtering for non-defects
        2. **NLP (PyThaiNLP):** 'newmm' tokenization & stopword removal
        3. **Feature Extraction:** TF-IDF (Unigrams & Bigrams)
        4. **Algorithm:** Nearest Centroid using Cosine Similarity
        """)
        
    # --- Backend Initialization ---
    with st.spinner("Initializing models..."):
        vectorizer, models, blacklist = train_models()
        
    if not models:
        st.error("Failed to load or train models.")
        st.stop()
        
    # --- UI TABS ---
    tab_classify, tab_evaluate = st.tabs(["Single Prediction", "Model Evaluation"])
    
    with tab_classify:
        # --- Main Input Section ---
        st.subheader("Input Defect Report")
        issue_text = st.text_area(
            "Paste or type a Thai software defect report here:", 
            height=150, 
            placeholder="e.g. แอพค้าง"
        )
        
        # --- Action Button ---
        analyze_btn = st.button("Analyze & Classify Defect", type="primary", use_container_width=True)
        
        # --- Result Section ---
        if analyze_btn:
            if not issue_text.strip():
                st.warning("Please enter some text to analyze.")
            else:
                # 1. Apply Blacklist Filtering
                is_task = any(word in issue_text.lower() for word in blacklist)
                    
                if is_task:
                    st.error("**Classification Rejected: Non-Defect Detected**")
                    st.markdown("This report has been identified as a **Task/Enhancement** based on rule-based blacklist filtering.")
                    st.info(f"Triggered by one of the blacklisted keywords: `{', '.join(blacklist)}`")
                else:
                    # 2. NLP Preprocessing Display
                    tokens = preprocess_text(issue_text)
                    st.subheader("1. NLP Preprocessing Results")
                    if not tokens:
                        st.warning("No meaningful tokens extracted after preprocessing. Try providing a more detailed description.")
                    else:
                        st.markdown("**Tokens extracted:**")
                        st.markdown(" ".join([f"`{t}`" for t in tokens]))
                        
                        # 3. Feature Extraction & TF-IDF Display
                        X_test = vectorizer.transform([issue_text])
                        
                        if X_test.nnz > 0:
                            st.markdown("**TF-IDF Vectorization Scores (Keyword Weights):**")
                            feature_names = vectorizer.get_feature_names_out()
                            tfidf_scores = zip(feature_names[X_test.nonzero()[1]], X_test.data)
                            sorted_tfidf = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
                            df_tfidf = pd.DataFrame(sorted_tfidf, columns=["Keyword", "TF-IDF Score"])
                            st.dataframe(df_tfidf.style.format({"TF-IDF Score": "{:.4f}"}), use_container_width=True)
                        
                        if X_test.nnz == 0:
                            st.warning("The input text contains no words known to the trained model. Unable to classify. Please provide a description with common defect terms.")
                        else:
                            predictions = {}
                            scores = {}
                            for dim, clf in models.items():
                                pred, score = clf.predict_with_score(X_test)
                                predictions[dim] = pred[0]
                                scores[dim] = score[0]
                                
                            # 4. Display Results in Metric Cards
                            st.subheader("2. Classification Results (IEEE 1044-2009)")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.success(f"### Defect Type\n**{predictions['type']}**\n\n*(Similarity: {scores['type']:.4f})*")
                                st.error(f"###  Severity Level\n**{predictions['severity']}**\n\n*(Similarity: {scores['severity']:.4f})*")
                                
                            with col2:
                                st.info(f"### Defect Effect\n**{predictions['effect']}**\n\n*(Similarity: {scores['effect']:.4f})*")
                                st.warning(f"### Defect Mode\n**{predictions['mode']}**\n\n*(Similarity: {scores['mode']:.4f})*")

    with tab_evaluate:
        st.subheader("80/20 Train-Test Evaluation Dashboard")
        st.markdown("This section automatically splits the dataset, trains the model on 80% of the data, and tests its predictions on the remaining 20% unseen data.")
        
        if st.button("Run Full Evaluation Pipeline", type="primary"):
            with st.spinner("Training models and running evaluation..."):
                metrics_data, train_size, test_size = evaluate_models()
                
                st.success(f"Evaluation complete! Trained on **{train_size}** reports and tested on **{test_size}** unseen reports.")
                
                # Dimension context: human-readable labels and academic explanations
                DIM_META = {
                    "type": {
                        "label": "Defect Type",
                        "description": "แยกประเภทของ Defects ตามลักษณะรหัส เช่น Logic, Interface, Data, Documentation",
                        "why_hard": "คำที่ใช้บรรยาย Defects Logic และ Interface มักปนกัน เพราะผู้แจ้งอธิบายจาก 'สิ่งที่เห็นบนหน้าจอ' ซึ่งอาจเป็น UI symptom ของ bug ประเภท Data ก็ได้"
                    },
                    "effect": {
                        "label": "Defect Effect",
                        "description": "ระบุผลกระทบที่เกิดขึ้น เช่น Functionality, Security, Performance",
                        "why_hard": "คลาส Performance มักมีข้อมูลน้อย (Class Imbalance) ทำให้ Centroid ไม่แข็งแรง จึงทายผิดบ่อย ส่วน Security มีคำเฉพาะชัดเจน จึงทายได้ดีกว่า"
                    },
                    "severity": {
                        "label": "Severity Level",
                        "description": "ระบุระดับความรุนแรงของ Defects เช่น Critical, Major, Minor",
                        "why_hard": "ระดับความรุนแรงขึ้นอยู่กับ 'บริบทของหน้าจอที่เกิดปัญหา' ไม่ใช่แค่ 'คำที่ใช้อธิบาย' ดังนั้น TF-IDF ซึ่งนับแค่คำจึงแยก Major-Minor ได้ยากมาก"
                    },
                    "mode": {
                        "label": "Defect Mode",
                        "description": "ระบุลักษณะการเกิดDefects เช่น Missing, Wrong, Extra",
                        "why_hard": "คำอย่าง 'ไม่พบ', 'หายไป' (Missing) กับ 'ผิด', 'เพี้ยน' (Wrong) มีความต่างชัดเจน ทำให้โมเดลแยกได้ดีที่สุดในทุก Dimension"
                    },
                }

                for dim, data in metrics_data.items():
                    st.divider()
                    meta = DIM_META.get(dim, {"label": dim.upper(), "description": "", "why_hard": ""})
                    st.markdown(f"### {meta['label']} (`{dim.upper()}`)")
                    st.caption(meta["description"])

                    acc = data['accuracy']
                    st.metric("Overall Accuracy", f"{acc * 100:.2f}%")

                    st.markdown("#### Classification Report")
                    report_df = pd.DataFrame(data['report']).transpose()
                    report_df = report_df.drop('accuracy', errors='ignore')
                    st.dataframe(report_df.style.format("{:.2f}"))

                    st.markdown("#### Confusion Matrix")
                    cm_df = pd.DataFrame(data['confusion_matrix'], index=data['classes'], columns=data['classes'])
                    st.dataframe(cm_df)

                    # --- Dynamic Summary Card ---
                    st.markdown("#### 🔍 Result Summary")

                    # Derive best/worst class from report (exclude macro/weighted avg)
                    class_rows = {
                        k: v for k, v in data['report'].items()
                        if k not in ('accuracy', 'macro avg', 'weighted avg')
                    }
                    if class_rows:
                        best_class = max(class_rows, key=lambda k: class_rows[k].get('f1-score', 0))
                        worst_class = min(class_rows, key=lambda k: class_rows[k].get('f1-score', 0))
                        best_f1 = class_rows[best_class]['f1-score']
                        worst_f1 = class_rows[worst_class]['f1-score']
                        worst_support = int(class_rows[worst_class].get('support', 0))

                        if acc >= 0.70:
                            overall_verdict = "✅ **ผลลัพธ์ดีมาก** — โมเดลสามารถแยกคลาสในมิตินี้ได้อย่างมีประสิทธิภาพ"
                            verdict_type = "success"
                        elif acc >= 0.50:
                            overall_verdict = "⚠️ **ผลลัพธ์ปานกลาง** — โมเดลยังทำงานได้บ้าง แต่มีความสับสนระหว่างบางคลาส"
                            verdict_type = "warning"
                        else:
                            overall_verdict = "❌ **ผลลัพธ์ต้องปรับปรุง** — โมเดลสับสนสูงมากในมิตินี้ ควรเพิ่มข้อมูลหรือปรับ Features"
                            verdict_type = "error"

                        summary_md = f"""
**คลาสที่ทำนายได้ดีที่สุด:** `{best_class}` — F1-Score: **{best_f1:.2f}**  
**คลาสที่ทำนายได้แย่ที่สุด:** `{worst_class}` — F1-Score: **{worst_f1:.2f}** (Support: {worst_support} ตัวอย่าง)

**สาเหตุที่คาดการณ์:** {meta['why_hard']}
"""
                        if verdict_type == "success":
                            st.success(overall_verdict)
                        elif verdict_type == "warning":
                            st.warning(overall_verdict)
                        else:
                            st.error(overall_verdict)
                        st.markdown(summary_md)


if __name__ == "__main__":
    main()
